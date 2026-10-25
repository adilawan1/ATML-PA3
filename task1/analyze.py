"""Task 1 tables and figures from saved results (every number traces to results/ or run logs).

Outputs in results/task1/report/:
  benchmark_table.{csv,md,tex}   7 conditions x 4 benchmarks, primary score and change from baseline (pp)
  resources.{csv,md}             trainable params, time, peak memory, checkpoint sizes, overheads
  regularization.json            Full FT vs L2-SP: Omega, lambda*Omega, displacement
  sft_loss.png, lr.png, l2sp_terms.png, benchmarks.png, diagnostics.png
"""
import argparse
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.utils import REPO_ROOT, load_config, load_json, out_root, read_jsonl, results_root, save_json

PRIMARY = [("gsm8k", "flexible_em", "GSM8K (flex EM)"), ("hellaswag", "acc_norm", "HellaSwag (acc_norm)"),
           ("boolq", "acc", "BoolQ (acc)"), ("arc_easy", "acc_norm", "ARC-Easy (acc_norm)")]
SECONDARY = [("gsm8k", "strict_em"), ("hellaswag", "acc"), ("arc_easy", "acc"), ("gsm8k", "truncation_rate"),
             ("gsm8k", "unextracted_rate"), ("gsm8k", "mean_gen_tokens")]
LABELS = {"baseline": "Baseline", "full_ft_mid": "Full FT (mid)", "full_ft": "Full FT", "full_ft_l2sp": "Full FT + L2-SP",
          "lora4": "LoRA-4", "lora16": "LoRA-16", "oplora16": "OPLoRA-16"}
VARIANTS = ["full_ft", "full_ft_l2sp", "lora4", "lora16", "oplora16"]


def smooth(x, w=25):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def benchmark_table(cfg, out):
    conds = [c["name"] for c in cfg["eval"]["conditions"]]
    S = {}
    for c in conds:
        p = results_root() / "task1/eval" / c / "scores.json"
        if p.exists():
            S[c] = load_json(p)
    if "baseline" not in S:
        print("no baseline scores yet")
        return
    rows = []
    for c in conds:
        if c not in S:
            continue
        r = {"condition": LABELS[c]}
        for task, met, lab in PRIMARY:
            v = S[c].get(task, {}).get(met)
            b = S["baseline"].get(task, {}).get(met)
            r[lab] = None if v is None else 100 * v
            r[f"Δ {lab} (pp)"] = None if v is None or b is None else 100 * (v - b)
        for task, met in SECONDARY:
            v = S[c].get(task, {}).get(met)
            r[f"{task}.{met}"] = (100 * v if v is not None and met != "mean_gen_tokens" else v)
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(out / "benchmark_table.csv", index=False, float_format="%.2f")
    (out / "benchmark_table.md").write_text(df.to_markdown(index=False, floatfmt=".2f"), encoding="utf-8")
    compact = df[["condition"] + [c for _, _, lab in PRIMARY for c in (lab, f"Δ {lab} (pp)")]]
    (out / "benchmark_table.tex").write_text(compact.to_latex(index=False, float_format="%.1f"), encoding="utf-8")

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    for ax, (_, _, lab) in zip(axes, PRIMARY):
        ax.bar(range(len(df)), df[f"Δ {lab} (pp)"], color=["#888"] + ["#4C72B0"] * (len(df) - 1))
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(df)), df["condition"], rotation=60, ha="right", fontsize=8)
        ax.set_title(lab, fontsize=9)
        ax.set_ylabel("Δ vs baseline (pp)")
    fig.tight_layout()
    fig.savefig(out / "benchmarks.png", dpi=200)
    print(df.to_markdown(index=False, floatfmt=".2f"))


def training_curves(out):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    for v in VARIANTS:
        p = out_root() / "task1/runs" / v / "train_log.jsonl"
        if not p.exists():
            continue
        log = read_jsonl(p)
        u = np.array([r["update"] for r in log])
        loss = [r["sft_loss"] for r in log]
        s = smooth(loss)
        ax.plot(u[len(u) - len(s):], s, label=LABELS[v])
        ax2.plot(u, [r["lr"] for r in log], label=LABELS[v])
        if v == "full_ft_l2sp":
            f3, a3 = plt.subplots(figsize=(6, 3))
            a3.plot(u, smooth(loss, 1), label="SFT loss", alpha=0.6)
            a3.plot(u, [r["lambda_omega"] for r in log], label="λΩ")
            a3.plot(u, [r["total_loss"] for r in log], label="total", alpha=0.6)
            a3.set_yscale("log")
            a3.set_xlabel("optimizer update")
            a3.legend()
            f3.tight_layout()
            f3.savefig(out / "l2sp_terms.png", dpi=200)
    ax.set_xlabel("optimizer update")
    ax.set_ylabel("SFT loss (25-update moving avg.)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "sft_loss.png", dpi=200)
    ax2.set_xlabel("optimizer update")
    ax2.set_ylabel("learning rate")
    ax2.set_yscale("log")
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(out / "lr.png", dpi=200)


def resources(out):
    rows, reg = [], {}
    for v in VARIANTS:
        p = out_root() / "task1/runs" / v / "summary.json"
        if not p.exists():
            continue
        s = load_json(p)
        cks = s.get("ckpt_sizes", {})
        sizes = cks[max(cks, key=int)] if cks else {}
        r = {"method": LABELS[v], "trainable_params_M": s["trainable_params"] / 1e6,
             "train_time_min": s["train_time_s"] / 60, "peak_alloc_GB": s["peak_alloc_gb"],
             "peak_reserved_GB": s["peak_reserved_gb"], "weights_ckpt_MB": sizes.get("model_bytes", 0) / 2**20,
             "recovery_ckpt_MB": sizes.get("total_bytes", 0) / 2**20, "skipped_updates": len(s["skipped_updates"]),
             "gpu": s.get("gpu"), "resumes": s.get("resumes", 0)}
        for k in ("basis_time_s", "basis_bytes", "export_time_s", "theta0_bytes", "theta0_copy_s", "penalty_time_s"):
            if k in s and s[k]:
                r[k] = s[k] / 2**20 if k.endswith("bytes") else s[k]
        rows.append(r)
        log = read_jsonl(out_root() / "task1/runs" / v / "train_log.jsonl")
        reg[v] = {"displacement": s.get("displacement"), "final_sft_loss_avg_last50": float(np.mean([r["sft_loss"] for r in log[-50:]]))}
        if v == "full_ft_l2sp":
            reg[v].update({"omega_final": log[-1]["omega"], "lambda_omega_final": log[-1]["lambda_omega"],
                           "lambda_omega_max": max(r["lambda_omega"] for r in log),
                           "lambda_omega_over_sft_final": log[-1]["lambda_omega"] / log[-1]["sft_loss"]})
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out / "resources.csv", index=False, float_format="%.3f")
        (out / "resources.md").write_text(df.to_markdown(index=False, floatfmt=".2f"), encoding="utf-8")
        print(df.to_markdown(index=False, floatfmt=".2f"))
    save_json(reg, out / "regularization.json")


def diagnostics_fig(out):
    p = results_root() / "task1/diagnostics/diagnostics.json"
    if not p.exists():
        return
    df = pd.DataFrame(load_json(p)["summary"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
    conds = list(dict.fromkeys(df["condition"]))
    w = 0.25
    for i, layer in enumerate(sorted(df["layer"].unique())):
        d = df[df.layer == layer].set_index("condition").loc[conds]
        x = np.arange(len(conds)) + (i - 1) * w
        axes[0].bar(x, d["rho_k"], w, label=f"layer {layer}")
        axes[1].bar(x, d["eta"], w, label=f"layer {layer}")
        axes[2].bar(x, d["R_ratio_math_over_nonmath"], w, label=f"layer {layer}")
    axes[0].axhline(df["rho_random_baseline"].iloc[0], ls="--", c="k", lw=0.8, label="random (k/d)")
    for ax, t in zip(axes, ["ρ_k (protected output subspace)", "η = ||ΔW||/||W0||", "mean R(z): math / non-math"]):
        ax.set_xticks(range(len(conds)), [LABELS[c] for c in conds], rotation=45, ha="right", fontsize=8)
        ax.set_title(t, fontsize=9)
    axes[1].set_yscale("log")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "diagnostics.png", dpi=200)


def sync_run_logs():
    """Copy small per-run logs from $PA3_OUT into results/ so they are committed with the repo."""
    for v in VARIANTS:
        src = out_root() / "task1/runs" / v
        for f in ("train_log.jsonl", "summary.json", "meta.json", "ckpt_sizes.json"):
            if (src / f).exists():
                (results_root() / "task1/runs" / v).mkdir(parents=True, exist_ok=True)
                shutil.copy(src / f, results_root() / "task1/runs" / v / f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs/task1.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = results_root() / "task1/report"
    out.mkdir(parents=True, exist_ok=True)
    sync_run_logs()
    training_curves(out)
    resources(out)
    benchmark_table(cfg, out)
    diagnostics_fig(out)


if __name__ == "__main__":
    main()
