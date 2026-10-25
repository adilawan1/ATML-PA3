"""Task 1 training: Full FT, Full FT + L2-SP, LoRA-4, LoRA-16, OPLoRA-16 (one script, resumable).

  python -m task1.train --variant full_ft
  python -m task1.train --variant oplora16          # re-running resumes from the newest checkpoint

Run directory: $PA3_OUT/task1/runs/<variant>/
  train_log.jsonl           one line per optimizer update
  ckpt_u{U}/                recovery checkpoint (model or adapter weights + state.pt)
  summary.json              resources, overheads, displacement, checkpoint sizes
"""
import argparse
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from common import lora as L
from common.utils import (REPO_ROOT, append_jsonl, count_params, dir_size_bytes, get_rng_state, gpu_info,
                          load_config, load_json, out_root, peak_mem_gb, save_json, set_rng_state, set_seed)


# ----------------------------------------------------------------------------- data
class TokenData:
    def __init__(self, path):
        z = np.load(path)
        self.ids, self.offsets, self.plen = z["ids"], z["offsets"], z["prompt_lens"]

    def __len__(self):
        return len(self.plen)

    def n_target_tokens(self, idx):
        # predicted tokens = response + <|im_end|> = length - prompt length
        return int(sum(self.offsets[i + 1] - self.offsets[i] - self.plen[i] for i in idx))

    def batch(self, idx, pad_id):
        seqs = [self.ids[self.offsets[i]:self.offsets[i + 1]] for i in idx]
        T = max(len(s) for s in seqs)
        input_ids = torch.full((len(idx), T), pad_id, dtype=torch.long)
        attn = torch.zeros((len(idx), T), dtype=torch.long)
        labels = torch.full((len(idx), T), -100, dtype=torch.long)
        for b, (i, s) in enumerate(zip(idx, seqs)):
            n = len(s)
            input_ids[b, :n] = torch.from_numpy(s.astype(np.int64))
            attn[b, :n] = 1
            labels[b, self.plen[i]:n] = input_ids[b, self.plen[i]:n]
        return input_ids, attn, labels


def sum_ce_loss(model, input_ids, attn, labels):
    """Summed next-token CE over supervised positions. LM head is applied only where labels exist."""
    h = model.model(input_ids=input_ids, attention_mask=attn).last_hidden_state
    tgt = labels[:, 1:]
    sel = tgt != -100
    logits = model.lm_head(h[:, :-1][sel])
    return F.cross_entropy(logits.float(), tgt[sel], reduction="sum")


# ----------------------------------------------------------------------------- model setup
def build_model(cfg, vcfg, device, model_name):
    method = vcfg["method"]
    # FP16 frozen backbone for adapters (FP32 on CPU smoke tests only: CPU FP16 matmuls are impractically slow)
    dtype = torch.float32 if method == "full" or device == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
    model.config.use_cache = False
    info = {}
    if method in ("lora", "oplora"):
        for p in model.parameters():
            p.requires_grad_(False)
        bases = None
        if method == "oplora":
            bases, binfo = L.compute_protection_bases(model, cfg["lora"]["targets"], vcfg["k"])
            info.update(binfo)
        L.inject_lora(model, cfg["lora"]["targets"], vcfg["r"], vcfg["alpha"], cfg["lora"]["dropout"],
                      cfg["seed"], bases)
        info["bases"] = bases
    if cfg["train"]["grad_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model, info


def param_groups(model, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():  # named_parameters de-duplicates tied weights
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 else no_decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


@torch.no_grad()
def displacement(model, theta0):
    sq = 0.0
    for p, p0 in zip(unique_params(model), theta0):
        sq += (p.detach().float() - p0.to(p.device)).pow(2).sum().item()
    return math.sqrt(sq)


def unique_params(model):
    return [p for _, p in model.named_parameters()]


# ----------------------------------------------------------------------------- checkpointing
def save_checkpoint(run_dir, u, model, method, opt, sched, scaler, state, keep_model):
    ck = run_dir / f"ckpt_u{u}"
    ck.mkdir(parents=True, exist_ok=True)
    if method == "full":
        model.save_pretrained(ck / "model", safe_serialization=True)  # FP32 master weights, tied stored once
    else:
        torch.save(L.adapter_state_dict(model), ck / "adapter_raw.pt")
        torch.save(L.export_effective_factors(model), ck / "adapter_effective.pt")
    torch.save({"optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                "rng": get_rng_state(), "update": u, "examples_seen": state["examples_seen"],
                "state": state}, ck / "state.pt")
    sizes = {"model_bytes": dir_size_bytes(ck / "model") if method == "full" else
             (ck / "adapter_raw.pt").stat().st_size,
             "state_bytes": (ck / "state.pt").stat().st_size, "keep_model": keep_model}
    if method != "full":
        sizes["effective_bytes"] = (ck / "adapter_effective.pt").stat().st_size
    sizes["total_bytes"] = dir_size_bytes(ck)
    return sizes


def prune_checkpoints(run_dir, keep_recovery, eval_updates, current_u):
    cks = sorted(run_dir.glob("ckpt_u*"), key=lambda p: int(p.name[6:]))
    old = [c for c in cks if int(c.name[6:]) != current_u]
    for c in old[: max(0, len(old) - (keep_recovery - 1))]:
        (c / "state.pt").unlink(missing_ok=True)  # drop optimizer state of older checkpoints
        if int(c.name[6:]) not in eval_updates and (c / "model").exists():
            shutil.rmtree(c / "model")


def latest_checkpoint(run_dir):
    cks = [c for c in run_dir.glob("ckpt_u*") if (c / "state.pt").exists()]
    return max(cks, key=lambda p: int(p.name[6:])) if cks else None


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs/task1.yaml"))
    ap.add_argument("--model", default=None, help="override model path (smoke tests)")
    ap.add_argument("--data", default=None)
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--max_updates", type=int, default=None, help="stop early (smoke tests)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tc, vcfg = cfg["train"], cfg["variants"][args.variant]
    method, lam = vcfg["method"], float(vcfg.get("l2sp", 0.0))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    run_dir = Path(args.run_dir) if args.run_dir else out_root() / "task1/runs" / args.variant
    run_dir.mkdir(parents=True, exist_ok=True)
    model_name = args.model or cfg["model"]
    data = TokenData(args.data or out_root() / "task1/data/train_tokens.npz")
    tok = AutoTokenizer.from_pretrained(model_name)

    set_seed(cfg["seed"])
    if amp:
        torch.cuda.reset_peak_memory_stats()
    t_setup = time.perf_counter()
    model, info = build_model(cfg, vcfg, device, model_name)
    bases = info.pop("bases", None)
    if bases is not None:
        torch.save(bases, run_dir / "oplora_bases.pt")
        info["basis_file_bytes"] = (run_dir / "oplora_bases.pt").stat().st_size
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(param_groups(model, tc["weight_decay"]), lr=vcfg["lr"], betas=tuple(tc["betas"]),
                            eps=tc["eps"])
    sched = get_cosine_schedule_with_warmup(opt, tc["warmup"], tc["updates"])
    scaler = torch.amp.GradScaler("cuda", init_scale=tc["init_scale"], enabled=amp)

    # starting point: GPU FP32 copy for L2-SP (penalty every update), CPU copy for Full FT displacement only
    theta0, overhead = None, {}
    if method == "full":
        t0 = time.perf_counter()
        theta0 = [p.detach().clone().float().to(device if lam > 0 else "cpu") for p in unique_params(model)]
        if amp:
            torch.cuda.synchronize()
        overhead["theta0_copy_s"] = time.perf_counter() - t0
        overhead["theta0_bytes"] = sum(t.numel() * 4 for t in theta0)
        overhead["theta0_device"] = "gpu" if lam > 0 else "cpu"

    state = {"examples_seen": 0, "train_time_s": 0.0, "penalty_time_s": 0.0,
             "skipped_updates": [], "peak_mem_segments": [], "displacement": {},
             "setup_time_s": time.perf_counter() - t_setup, "resumes": 0}
    start = 0
    ck = latest_checkpoint(run_dir)
    if ck is not None:
        s = torch.load(ck / "state.pt", map_location="cpu", weights_only=False)
        if method == "full":
            from safetensors.torch import load_file
            import glob
            sd = {}
            for f in glob.glob(str(ck / "model" / "*.safetensors")):
                sd.update(load_file(f))
            res = model.load_state_dict(sd, strict=False)
            assert not res.unexpected_keys and set(res.missing_keys) <= {"lm_head.weight"}, res  # tied head only
            model.tie_weights()
        else:
            L.load_adapter_state_dict(model, torch.load(ck / "adapter_raw.pt"))
        opt.load_state_dict(s["optimizer"])
        sched.load_state_dict(s["scheduler"])
        scaler.load_state_dict(s["scaler"])
        set_rng_state(s["rng"])
        state = s["state"]
        state["resumes"] += 1
        start = s["update"]
        print(f"resumed from {ck.name} (update {start})", flush=True)

    n_trainable = count_params(trainable)
    meta = {"variant": args.variant, "method": method, "lr": vcfg["lr"], "l2sp_lambda": lam,
            "trainable_params": n_trainable, "total_params": count_params(model.parameters()),
            **{k: v for k, v in vcfg.items() if k in ("r", "alpha", "k")}, **gpu_info(), **info, **overhead,
            "config": cfg}
    save_json(meta, run_dir / "meta.json")
    print(f"{args.variant}: trainable params {n_trainable:,}", flush=True)

    mb, ga = tc["micro_bs"], tc["grad_accum"]
    eff = mb * ga
    end = min(tc["updates"], args.max_updates) if args.max_updates else tc["updates"]
    ps = unique_params(model)
    for u in range(start, end):
        t_u = time.perf_counter()
        idx_all = list(range(u * eff, (u + 1) * eff))
        n_tok = data.n_target_tokens(idx_all)
        loss_sum = 0.0
        for m in range(ga):
            ids, attn, labels = data.batch(idx_all[m * mb:(m + 1) * mb], tok.pad_token_id)
            ids, attn, labels = ids.to(device), attn.to(device), labels.to(device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
                ls = sum_ce_loss(model, ids, attn, labels)
            scaler.scale(ls / n_tok).backward()
            loss_sum += ls.item()
        sft = loss_sum / n_tok
        scaler.unscale_(opt)
        omega = None
        if lam > 0:
            tp = time.perf_counter()
            with torch.no_grad():
                sq = torch.zeros((), device=device, dtype=torch.float32)
                for p, p0 in zip(ps, theta0):
                    d = p.detach() - p0
                    sq += d.pow(2).sum()
                    p.grad.add_(d, alpha=lam)  # d/dθ [λ/2 ||θ-θ0||²] = λ(θ-θ0); clipped jointly with SFT grad
                omega = 0.5 * sq.item()
            state["penalty_time_s"] += time.perf_counter() - tp
        gnorm = torch.nn.utils.clip_grad_norm_(trainable, tc["clip"]).item()
        lr = sched.get_last_lr()[0]
        scale_before = scaler.get_scale() if amp else 1.0
        scaler.step(opt)
        scaler.update()
        skipped = amp and scaler.get_scale() < scale_before
        sched.step()
        opt.zero_grad(set_to_none=True)
        state["examples_seen"] = (u + 1) * eff
        dt = time.perf_counter() - t_u
        state["train_time_s"] += dt
        if skipped:
            state["skipped_updates"].append(u + 1)
        rec = {"update": u + 1, "sft_loss": sft, "lr": lr, "grad_norm": gnorm, "n_tokens": n_tok,
               "scale": scaler.get_scale() if amp else 1.0, "skipped": bool(skipped), "step_time_s": dt}
        if lam > 0:
            rec.update({"omega": omega, "lambda_omega": lam * omega, "total_loss": sft + lam * omega})
        append_jsonl(rec, run_dir / "train_log.jsonl")
        if (u + 1) % 10 == 0 or u == start:
            print(f"[{args.variant}] u{u + 1} loss {sft:.4f} lr {lr:.2e} gn {gnorm:.2f}"
                  + (f" Ω {omega:.4g}" if omega is not None else "") + (" SKIPPED" if skipped else ""), flush=True)

        if (u + 1) in tc["ckpt_updates"]:
            tc0 = time.perf_counter()
            if method == "full" and (u + 1) in tc["eval_updates"]:
                state["displacement"][str(u + 1)] = displacement(model, theta0)
            state["peak_mem_segments"].append(peak_mem_gb())
            sizes = save_checkpoint(run_dir, u + 1, model, method, opt, sched, scaler, state,
                                    keep_model=(u + 1) in tc["eval_updates"])
            prune_checkpoints(run_dir, tc["keep_recovery"], tc["eval_updates"], u + 1)
            sizes["save_time_s"] = time.perf_counter() - tc0
            # kept outside state.pt so the entry for this checkpoint survives a resume from it
            sz_path = run_dir / "ckpt_sizes.json"
            all_sizes = load_json(sz_path) if sz_path.exists() else {}
            all_sizes[str(u + 1)] = sizes
            save_json(all_sizes, sz_path)

    if end == tc["updates"] and method != "full":
        # export overhead: time to materialize projected factors for inference
        te = time.perf_counter()
        eff_f = L.export_effective_factors(model)
        state["export_time_s"] = time.perf_counter() - te
        state["export_bytes"] = sum(v.numel() * v.element_size() for v in eff_f.values())
    state["peak_mem_segments"].append(peak_mem_gb())
    summary = {**{k: v for k, v in meta.items() if k != "config"}, **state,
               "peak_alloc_gb": max(s["peak_alloc_gb"] for s in state["peak_mem_segments"]),
               "peak_reserved_gb": max(s["peak_reserved_gb"] for s in state["peak_mem_segments"]),
               "completed_updates": end,
               "ckpt_sizes": load_json(run_dir / "ckpt_sizes.json") if (run_dir / "ckpt_sizes.json").exists() else {}}
    save_json(summary, run_dir / "summary.json")
    print("done:", {k: summary[k] for k in ("train_time_s", "peak_alloc_gb", "trainable_params")}, flush=True)


if __name__ == "__main__":
    main()
