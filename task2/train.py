"""Three-stage VLM training (caption alignment -> visual instruction tuning -> natural VQA), resumable.

  python -m task2.train                 # 196 visual tokens (main run)
  python -m task2.train --n_vis 49      # optional 2x2-pooled run (new connector/adapters, same everything else)

Run dir $PA3_OUT/task2/runs/<run>/ : train_log.jsonl, dev_log.jsonl, ckpt_u{U}/ (every 250 updates),
stage{k}_weights.pt (connector + adapters after each stage), summary.json
"""
import argparse
import shutil
import time
from collections import Counter

import numpy as np
import torch

from common import lora as L
from common.utils import (REPO_ROOT, append_jsonl, atomic_torch_save, count_params, get_rng_state, gpu_info,
                          load_config, out_root, peak_mem_gb, results_root, save_json, set_rng_state, set_seed,
                          truncate_jsonl)
from task2.data import load_all, pair_examples
from task2.vlm import FeatureStore, add_lora, build_vlm, calibrate_scale, lr_lambda


# ----------------------------------------------------------------------------- sampling
class Sampler:
    """Seeded sampler. Stage 1: two shuffled passes over the caption examples. Stages 2-3: each draw picks a
    component by its mixture probability and contributes two examples (a full question pair, or two captions)."""

    def __init__(self, data, seed):
        self.D, self.seed = data, seed
        self.rng = np.random.default_rng(seed + 1)
        self.vqa_pairs = data["vqa_train"]
        self.pools = {"presence": data["pool_presence"], "counting": data["pool_counting"]}

    def stage1_batch(self, step, n):
        caps = self.D["captions_train"]
        per_epoch = len(caps) // n
        epoch, pos = divmod(step, per_epoch)
        perm = np.random.default_rng(self.seed + 100 + epoch).permutation(len(caps))
        return [caps[i] for i in perm[pos * n:(pos + 1) * n]], Counter({"caption": n})

    def draw(self, mixture, n_examples):
        names, probs = list(mixture), np.array(list(mixture.values()))
        out, counts = [], Counter()
        for _ in range(n_examples // 2):
            k = names[self.rng.choice(len(names), p=probs / probs.sum())]
            counts[k] += 1
            if k == "caption":
                caps = self.D["captions_train"]
                out += [caps[i] for i in self.rng.integers(0, len(caps), 2)]
            else:
                src = self.vqa_pairs if k == "vqa" else self.pools[k]
                out += pair_examples([src[self.rng.integers(len(src))]], "train2014")
        return out, counts


# ----------------------------------------------------------------------------- dev evaluation
@torch.no_grad()
def dev_eval(vlm, store, data, bs=16):
    vlm.eval()
    amp = vlm.device.type == "cuda"
    res = {}

    # caption reference loss (training loss definition: caption tokens + end-of-turn)
    caps = data["captions_dev"]
    vals = []
    for i in range(0, len(caps), bs):
        chunk = caps[i:i + bs]
        ids, lab, att, f, h = vlm.collate(chunk, store.get, [True] * len(chunk), label_eot=True)
        with torch.autocast(device_type=vlm.device.type, dtype=torch.float16, enabled=amp):
            vals += vlm.per_example_nll(ids, lab, att, f, h).tolist()
    res["caption_loss"] = float(np.mean(vals))
    # reference-answer NLL (termination excluded) under correct / complementary / no image
    for suite in ("grounding_dev", "vqa_dev"):
        exs = pair_examples(data[suite], "val2014")
        partner = [dict(e, show_image=exs[i ^ 1]["image_id"]) for i, e in enumerate(exs)]  # slots 0/1 adjacent
        r = {}
        for cond, xs, wi in (("correct", exs, True), ("complementary", partner, True), ("none", exs, False)):
            v = []
            for i in range(0, len(xs), bs):
                chunk = xs[i:i + bs]
                ids, lab, att, f, h = vlm.collate(chunk, store.get, [wi] * len(chunk), label_eot=False)
                with torch.autocast(device_type=vlm.device.type, dtype=torch.float16, enabled=amp):
                    v += vlm.per_example_nll(ids, lab, att, f, h).tolist()
            r[cond] = np.array(v)
        res[suite] = {f"nll_{c}": float(v.mean()) for c, v in r.items()}
        res[suite]["swap_penalty"] = float((r["complementary"] - r["correct"]).mean())
        for qt in sorted({e["qtype"] for e in exs}):
            m = np.array([e["qtype"] == qt for e in exs])
            res[suite][f"swap_penalty_{qt}"] = float((r["complementary"][m] - r["correct"][m]).mean())
            res[suite][f"nll_correct_{qt}"] = float(r["correct"][m].mean())
    vlm.train()
    return res


# ----------------------------------------------------------------------------- main
def param_groups(vlm, stage, wd):
    conn = [(n, p) for n, p in vlm.connector.named_parameters()]
    groups = [{"params": [p for n, p in conn if p.ndim >= 2], "weight_decay": wd, "lr": stage["connector_lr"], "name": "connector"},
              {"params": [p for n, p in conn if p.ndim < 2], "weight_decay": 0.0, "lr": stage["connector_lr"], "name": "connector_bias"}]
    if stage["lora_lr"]:
        lp = [p for m in L.lora_modules(vlm.llm).values() for p in (m.A, m.B)]
        groups.append({"params": lp, "weight_decay": wd, "lr": stage["lora_lr"], "name": "lora"})
    return groups


def _norm(gs):
    return float(sum((g.float().norm() ** 2).item() for g in gs if g is not None) ** 0.5)


def interface_checks(vlm, trainable):
    """Frozen-parameter sets and gradient flow (run after unscaling, before the optimizer step).

    Non-finite gradients are not a failure here: with FP16 dynamic loss scaling the first updates can overflow,
    and the GradScaler then skips the update and lowers the scale. The caller repeats the check on the next
    update until the gradients are finite, so gradient flow is always verified on a real update.
    """
    ids = {id(p) for p in trainable}
    lora_ids = {id(p) for m in L.lora_modules(vlm.llm).values() for p in (m.A, m.B)}
    backbone = [p for p in vlm.llm.parameters() if id(p) not in lora_ids]
    conn_g = [p.grad for p in vlm.connector.parameters()]
    lora_g = [p.grad for m in L.lora_modules(vlm.llm).values() for p in (m.A, m.B)]
    out = {"backbone_params_requiring_grad": sum(p.requires_grad for p in backbone),
           "backbone_params_with_grad": sum(p.grad is not None for p in backbone),
           "trainable_tensors_in_optimizer": len(ids),
           "connector_grad_norm": _norm(conn_g),
           "connector_params_without_grad": sum(g is None for g in conn_g),
           "lora_grad_norm": _norm(lora_g),
           "lora_params_without_grad": sum(g is None for g in lora_g),
           "lora_tensors": len(lora_g)}
    out["grads_finite"] = bool(np.isfinite(out["connector_grad_norm"]) and np.isfinite(out["lora_grad_norm"]))
    problems = []
    if out["backbone_params_requiring_grad"] or out["backbone_params_with_grad"]:
        problems.append("frozen LLM parameters require or received gradients")
    if out["connector_params_without_grad"]:
        problems.append("some connector parameters received no gradient")
    if lora_g and out["lora_params_without_grad"]:
        problems.append("some LoRA parameters received no gradient")
    if out["grads_finite"] and out["connector_grad_norm"] == 0:
        problems.append("connector gradient is exactly zero (no gradient flow from the loss to the visual tokens)")
    if problems:
        raise RuntimeError(f"interface check failed: {problems}; values: {out}")
    return out


def save_ckpt(path, vlm, opt, sched, scaler, sampler, st):
    path.mkdir(parents=True, exist_ok=True)
    atomic_torch_save({"connector": vlm.connector.state_dict(), "lora": L.adapter_state_dict(vlm.llm),
                       "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                       "rng": get_rng_state(), "sampler_rng": sampler.rng.bit_generator.state, "state": st},
                      path / "state.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs/task2.yaml"))
    ap.add_argument("--n_vis", type=int, default=None)
    ap.add_argument("--run", default=None)
    ap.add_argument("--max_updates", type=int, default=None, help="smoke tests only")
    args = ap.parse_args()
    cfg = load_config(args.config)
    tc = cfg["train"]
    n_vis = args.n_vis or cfg["n_visual_tokens"]
    run = args.run or f"vis{n_vis}"
    rd = out_root() / "task2/runs" / run
    rd.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    set_seed(cfg["seed"])
    if amp:
        torch.cuda.reset_peak_memory_stats()
    data = load_all()
    store = FeatureStore(n_vis)
    vlm, tok = build_vlm(cfg, device, n_vis)
    if tc["grad_checkpointing"]:
        vlm.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    sampler = Sampler(data, cfg["seed"])
    stages = cfg["train"]["stages"]
    bounds = np.cumsum([0] + [s["updates"] for s in stages])
    total = int(bounds[-1])
    end = min(total, args.max_updates) if args.max_updates else total
    eff = tc["micro_bs"] * tc["grad_accum"]

    # resume
    cks = sorted((c for c in rd.glob("ckpt_u*") if (c / "state.pt").exists()), key=lambda p: int(p.name[6:]))
    st = {"update": 0, "stage_time_s": {}, "stage_peak_mem": {}, "mixture_counts": {}, "skipped": [],
          "trainable_params": {}, "dev_done": []}
    resume = torch.load(cks[-1] / "state.pt", map_location="cpu", weights_only=False) if cks else None
    if resume is None:
        cal = calibrate_scale(vlm, store, data["captions_train"] + pair_examples(data["vqa_train"], "train2014"),
                              seed=cfg["seed"])
        save_json(cal, results_root() / "task2" / run / "calibration.json")
        st["calibration"] = cal
        # fresh start: discard log rows of an earlier attempt that never reached a checkpoint
        for f in ("train_log.jsonl", "dev_log.jsonl"):
            truncate_jsonl(rd / f, "update", -1)
    else:
        st = resume["state"]
        vlm.connector.load_state_dict(resume["connector"])
    stage_idx = int(np.searchsorted(bounds, st["update"], side="right") - 1)
    if resume and resume["lora"]:
        add_lora(vlm, cfg)
        L.load_adapter_state_dict(vlm.llm, resume["lora"])

    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    pending_opt = None
    if resume is not None:
        scaler.load_state_dict(resume["scaler"])
        set_rng_state(resume["rng"])
        sampler.rng.bit_generator.state = resume["sampler_rng"]
        pending_opt = (resume["optimizer"], resume["scheduler"])
        dropped = truncate_jsonl(rd / "train_log.jsonl", "update", st["update"])
        dropped += truncate_jsonl(rd / "dev_log.jsonl", "update", st["update"])
        print(f"resumed at update {st['update']}; dropped {dropped} log lines past the checkpoint", flush=True)
    opt = sched = None
    run_skips = 0
    vlm.train()
    u = st["update"]
    while u < end:
        k = int(np.searchsorted(bounds, u, side="right") - 1)
        stage = stages[k]
        s_step = u - int(bounds[k])
        if opt is None or s_step == 0:
            if stage["lora_lr"] and not L.lora_modules(vlm.llm):
                add_lora(vlm, cfg)
            opt = torch.optim.AdamW(param_groups(vlm, stage, tc["weight_decay"]), betas=(0.9, 0.999), eps=1e-8)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(stage["updates"], tc["warmup_frac"], tc["min_lr_frac"]))
            trainable = [p for g in opt.param_groups for p in g["params"]]
            st["trainable_params"][stage["name"]] = count_params(trainable)
            if pending_opt is not None and s_step != 0:
                opt.load_state_dict(pending_opt[0])
                sched.load_state_dict(pending_opt[1])
            pending_opt = None
            if amp:
                torch.cuda.reset_peak_memory_stats()
        if u == 0 and 0 not in st["dev_done"]:
            dv = dev_eval(vlm, store, data)
            append_jsonl({"update": 0, "stage": "init", **dv}, rd / "dev_log.jsonl")
            st["dev_done"].append(0)
        t0 = time.perf_counter()
        if k == 0:
            exs, counts = sampler.stage1_batch(s_step, eff)
        else:
            exs, counts = sampler.draw(stage["mixture"], eff)
        loss_tot = 0.0
        for m in range(tc["grad_accum"]):
            chunk = exs[m * tc["micro_bs"]:(m + 1) * tc["micro_bs"]]
            ids, lab, att, f, h = vlm.collate(chunk, store.get, [True] * len(chunk), label_eot=True)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
                per_ex = vlm.per_example_nll(ids, lab, att, f, h)
            loss = per_ex.sum() / eff  # example-weighted: token mean within example, mean over effective batch
            scaler.scale(loss).backward()
            loss_tot += loss.item()
        scaler.unscale_(opt)
        if stage["name"] not in st.get("checks", {}):
            chk = interface_checks(vlm, trainable)
            if chk["grads_finite"]:
                chk["verified_at_update"] = u + 1
                st.setdefault("checks", {})[stage["name"]] = chk
                print("checks:", chk, flush=True)
            else:
                print(f"u{u + 1} [{stage['name']}]: non-finite gradients (FP16 overflow, loss {loss_tot:.4f}); the "
                      f"loss scaler skips this update and lowers the scale; interface check repeated next update",
                      flush=True)
        gn = torch.nn.utils.clip_grad_norm_(trainable, tc["clip"]).item()
        lrs = {g["name"]: g["lr"] for g in opt.param_groups}
        sb = scaler.get_scale() if amp else 1.0
        scaler.step(opt)
        scaler.update()
        skipped = amp and scaler.get_scale() < sb
        sched.step()
        opt.zero_grad(set_to_none=True)
        u += 1
        dt = time.perf_counter() - t0
        st["update"] = u
        st["stage_time_s"][stage["name"]] = st["stage_time_s"].get(stage["name"], 0.0) + dt
        mc = st["mixture_counts"].setdefault(stage["name"], {})
        for kk, v in counts.items():
            mc[kk] = mc.get(kk, 0) + v
        if skipped:
            st["skipped"].append(u)
        run_skips = run_skips + 1 if skipped else 0
        if run_skips >= 30:
            raise RuntimeError(f"{run_skips} consecutive skipped updates (non-finite gradients) up to update {u}, "
                               f"last loss {loss_tot}: training is not progressing in FP16")
        append_jsonl({"update": u, "stage": stage["name"], "loss": loss_tot, "grad_norm": gn, "lr": lrs,
                      "skipped": bool(skipped), "draws": dict(counts), "step_time_s": dt}, rd / "train_log.jsonl")
        if u % 50 == 0 or u == 1:
            print(f"u{u} [{stage['name']}] loss {loss_tot:.4f} gn {gn:.2f} lr {lrs}", flush=True)
        st["stage_peak_mem"][stage["name"]] = max(st["stage_peak_mem"].get(stage["name"], 0.0), peak_mem_gb()["peak_alloc_gb"])
        at_stage_end = u in bounds[1:]
        if (u % tc["dev_eval_every"] == 0 or at_stage_end) and u not in st["dev_done"]:
            dv = dev_eval(vlm, store, data)
            append_jsonl({"update": u, "stage": stage["name"], **dv}, rd / "dev_log.jsonl")
            st["dev_done"].append(u)
            print("dev", u, {k2: (round(v, 3) if isinstance(v, float) else v) for k2, v in dv.items() if k2 == "caption_loss"},
                  dv["grounding_dev"]["swap_penalty"], dv["vqa_dev"]["swap_penalty"], flush=True)
        if at_stage_end:
            torch.save({"connector": vlm.connector.state_dict(), "lora": L.adapter_state_dict(vlm.llm)},
                       rd / f"stage{k + 1}_weights.pt")
        if u % tc["ckpt_every"] == 0 or u == end:
            save_ckpt(rd / f"ckpt_u{u}", vlm, opt, sched, scaler, sampler, st)
            for old in sorted(rd.glob("ckpt_u*"), key=lambda p: int(p.name[6:]))[:-1]:
                shutil.rmtree(old, ignore_errors=True)

    summary = {"run": run, "n_visual_tokens": n_vis, **gpu_info(), **{k: v for k, v in st.items() if k != "dev_done"},
               "total_train_time_s": sum(st["stage_time_s"].values()), "completed_updates": u,
               "example_presentations": u * eff}
    save_json(summary, rd / "summary.json")
    rr = results_root() / "task2" / run
    rr.mkdir(parents=True, exist_ok=True)
    for f in ("train_log.jsonl", "dev_log.jsonl", "summary.json"):
        if (rd / f).exists():
            (rr / f).write_bytes((rd / f).read_bytes())
    print("done", {k: summary[k] for k in ("total_train_time_s", "trainable_params", "stage_peak_mem")})


if __name__ == "__main__":
    main()
