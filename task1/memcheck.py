"""GPU memory check for one Task 1 variant on its worst micro-batches (no checkpoints written, nothing trained).

Builds the model and optimizer exactly as task1.train does, then runs two full updates on the micro-batches of
the real training schedule with (a) the most supervised tokens and (b) the most padded tokens, so the Adam
state is allocated and the peak includes the largest LM-head logits. Learning rate is 0 at update 1, so the
throwaway model is not even changed. Prints and writes one JSON record.

  python -m task1.memcheck --variant full_ft_l2sp [--loss_chunk 1024] [--theta0_device cpu] --out m.json
"""
import argparse
import json

import torch

from common.utils import REPO_ROOT, load_config, out_root, set_seed
from task1.train import TokenData, build_model, param_groups, sum_ce_loss, unique_params


def worst_microbatches(data, updates, mb, ga):
    best_s, best_t = (-1, None), (-1, None)
    for u in range(updates):
        for m in range(ga):
            idx = list(range(u * mb * ga + m * mb, u * mb * ga + (m + 1) * mb))
            lens = [int(data.offsets[i + 1] - data.offsets[i]) for i in idx]
            s = sum(n - int(data.plen[i]) for n, i in zip(lens, idx))
            t = max(lens) * len(idx)
            best_s = max(best_s, (s, idx))
            best_t = max(best_t, (t, idx))
    return {"most_supervised": best_s, "most_padded": best_t}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs/task1.yaml"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--loss_chunk", type=int, default=0)
    ap.add_argument("--theta0_device", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    tc, vcfg = cfg["train"], cfg["variants"][args.variant]
    lam = float(vcfg.get("l2sp", 0.0))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    data = TokenData(args.data or out_root() / "task1/data/train_tokens.npz")
    worst = worst_microbatches(data, tc["updates"], tc["micro_bs"], tc["grad_accum"])
    rec = {"variant": args.variant, "loss_chunk": args.loss_chunk, "theta0_device": args.theta0_device,
           "worst_microbatches": {k: {"tokens": v[0], "indices": v[1]} for k, v in worst.items()}}
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model or cfg["model"])
    set_seed(cfg["seed"])
    try:
        model, _ = build_model(cfg, vcfg, device, args.model or cfg["model"])
        model.train()
        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(param_groups(model, tc["weight_decay"]), lr=0.0)
        scaler = torch.amp.GradScaler("cuda", init_scale=tc["init_scale"], enabled=amp)
        theta0 = None
        if lam > 0:
            dev0 = device if args.theta0_device == "gpu" else "cpu"
            theta0 = [p.detach().clone().float().to(dev0) for p in unique_params(model)]
        if amp:
            torch.cuda.reset_peak_memory_stats()
        for idx in (worst["most_supervised"][1], worst["most_padded"][1], worst["most_supervised"][1]):
            ids, attn, labels = (t.to(device) for t in data.batch(idx, tok.pad_token_id))
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
                ls = sum_ce_loss(model, ids, attn, labels, args.loss_chunk)
            scaler.scale(ls / max(1, int((labels != -100).sum()))).backward()
            scaler.unscale_(opt)
            if theta0 is not None:
                with torch.no_grad():
                    for p, p0 in zip(unique_params(model), theta0):
                        p.grad.add_(p.detach() - p0.to(p.device), alpha=lam)
            torch.nn.utils.clip_grad_norm_(trainable, tc["clip"])
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        rec["ok"] = True
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" not in str(e).lower() and not isinstance(e, torch.cuda.OutOfMemoryError):
            raise
        rec["ok"], rec["error"] = False, str(e).splitlines()[0]
    if amp:
        total = torch.cuda.get_device_properties(0).total_memory
        rec.update({"gpu": torch.cuda.get_device_name(0), "total_gb": total / 2**30,
                    "peak_alloc_gb": torch.cuda.max_memory_allocated() / 2**30,
                    "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30})
        rec["headroom_ok"] = rec["ok"] and torch.cuda.max_memory_reserved() <= 0.92 * total
    else:
        rec["headroom_ok"] = rec["ok"]
    with open(args.out, "w") as f:
        json.dump(rec, f, indent=2)
    print(json.dumps({k: v for k, v in rec.items() if k != "worst_microbatches"}))


if __name__ == "__main__":
    main()
