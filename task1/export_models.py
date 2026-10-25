"""Build FP16 Hugging Face model directories for every evaluation condition.

Full FT: FP32 master weights -> FP16.  LoRA/OPLoRA: merge exported (projected) factors into a *fresh*
FP16 backbone, then compare merged vs. unmerged logits on a small batch. For OPLoRA also verify the
orthogonality constraints numerically.

  python -m task1.export_models --condition lora16 --out_dir /content/merged
Writes results/task1/export/<condition>.json
"""
import argparse
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import lora as L
from common.utils import REPO_ROOT, dir_size_bytes, load_config, out_root, results_root, save_json

CHECK_PROMPTS = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did she sell altogether?",
    "hey I feel bored",
    "A man is sitting on a roof. He",
    "Which gas do plants absorb from the air for photosynthesis?",
]


def condition_cfg(cfg, name):
    return next(c for c in cfg["eval"]["conditions"] if c["name"] == name)


def run_dir(variant):
    return out_root() / "task1/runs" / variant


@torch.no_grad()
def check_logits(merged, unmerged, tok, device):
    tok.padding_side = "left"
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False) for p in CHECK_PROMPTS]
    enc = tok(texts, return_tensors="pt", padding=True).to(device)
    amp = device == "cuda"
    with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
        a = merged(**enc).logits.float()
        b = unmerged(**enc).logits.float()
    m = enc["attention_mask"].bool()
    a, b = a[m], b[m]
    return {"max_abs_diff": (a - b).abs().max().item(), "mean_abs_diff": (a - b).abs().mean().item(),
            "argmax_agreement": (a.argmax(-1) == b.argmax(-1)).float().mean().item(),
            "n_positions": int(m.sum())}


@torch.no_grad()
def oplora_constraint_check(model):
    worst_u, worst_v = 0.0, 0.0
    for _, m in L.lora_modules(model).items():
        A, B = m.effective_factors()
        if B.norm() > 0:
            worst_u = max(worst_u, ((m.U_k.T @ B).norm() / B.norm()).item())
        worst_v = max(worst_v, ((A @ m.V_k).norm() / A.norm()).item())
    return {"max_rel_UkT_Btilde": worst_u, "max_rel_Atilde_Vk": worst_v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs/task1.yaml"))
    ap.add_argument("--out_dir", required=True, help="where merged FP16 dirs go (local scratch is fine)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--skip_check", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cond = condition_cfg(cfg, args.condition)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_name = args.model or cfg["model"]
    tok = AutoTokenizer.from_pretrained(base_name)
    dst = Path(args.out_dir) / args.condition
    rep = {"condition": args.condition}
    t0 = time.perf_counter()

    if cond["variant"] is None:
        model = AutoModelForCausalLM.from_pretrained(base_name, dtype=torch.float16)
    else:
        vcfg = cfg["variants"][cond["variant"]]
        ck = run_dir(cond["variant"]) / f"ckpt_u{cond['update']}"
        if vcfg["method"] == "full":
            model = AutoModelForCausalLM.from_pretrained(ck / "model", dtype=torch.float32).half()
        else:
            factors = torch.load(ck / "adapter_effective.pt")
            model = AutoModelForCausalLM.from_pretrained(base_name, dtype=torch.float16).to(device)
            L.merge_factors_into(model, factors, dtype=torch.float16)
            rep["merge_time_s"] = time.perf_counter() - t0
            if not args.skip_check:
                # unmerged reference: fresh FP16 backbone + adapter modules loaded from the raw factors
                ref = AutoModelForCausalLM.from_pretrained(base_name, dtype=torch.float16).to(device)
                bases = None
                if vcfg["method"] == "oplora":
                    bases = torch.load(run_dir(cond["variant"]) / "oplora_bases.pt")
                L.inject_lora(ref, cfg["lora"]["targets"], vcfg["r"], vcfg["alpha"], cfg["lora"]["dropout"],
                              cfg["seed"], bases)
                L.load_adapter_state_dict(ref, torch.load(ck / "adapter_raw.pt"))
                ref.eval()
                model.eval()
                rep["logit_check"] = check_logits(model.to(device), ref, tok, device)
                if bases is not None:
                    rep["oplora_constraints"] = oplora_constraint_check(ref)
                del ref
    model.config.use_cache = True
    model.save_pretrained(dst, safe_serialization=True)
    tok.save_pretrained(dst)
    rep["export_total_time_s"] = time.perf_counter() - t0
    rep["merged_dir_bytes"] = dir_size_bytes(dst)
    rep["path"] = str(dst)
    save_json(rep, results_root() / "task1/export" / f"{args.condition}.json")
    print(rep)


if __name__ == "__main__":
    main()
