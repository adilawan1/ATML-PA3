"""Task 2 figures and tables.

  python -m task2.analyze [--run vis196] [--compare vis49]
Outputs in results/task2/<run>/report/: curves.png, final_table.{md,tex}, by_type.md, advantages.md,
resources.md, qualitative.png (if COCO images are available locally), compression.md (optional run)
"""
import argparse
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.utils import REPO_ROOT, load_config, load_json, read_jsonl, results_root

CONDS = ["correct", "complementary", "mismatched", "none"]
SUITES = {"grounding_test": "Annotation-derived grounding", "vqa_test": "Natural VQAv2"}


def curves(run, out, cfg):
    tl = read_jsonl(results_root() / "task2" / run / "train_log.jsonl")
    dl = read_jsonl(results_root() / "task2" / run / "dev_log.jsonl")
    bounds = np.cumsum([s["updates"] for s in cfg["train"]["stages"]])[:-1]
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.4))
    u = np.array([r["update"] for r in tl])
    loss = np.array([r["loss"] for r in tl])
    w = min(25, len(loss))
    axes[0].plot(u, loss, alpha=0.25, lw=0.6)
    axes[0].plot(u[w - 1:], np.convolve(loss, np.ones(w) / w, mode="valid"), lw=1.2)
    axes[0].set_title("training loss (example-weighted)", fontsize=9)
    du = [r["update"] for r in dl]
    axes[1].plot(du, [r["caption_loss"] for r in dl], "o-", ms=3)
    axes[1].set_title("dev caption reference loss", fontsize=9)
    for s, ls in (("grounding_dev", "-"), ("vqa_dev", "--")):
        for c, col in (("correct", "C0"), ("complementary", "C1"), ("none", "C2")):
            axes[2].plot(du, [r[s][f"nll_{c}"] for r in dl], ls, c=col, marker="o", ms=2,
                         label=f"{s.split('_')[0]}: {c}")
        axes[3].plot(du, [r[s]["swap_penalty"] for r in dl], ls, marker="o", ms=3, label=s.split("_")[0])
        for k in [k for k in dl[-1][s] if k.startswith("swap_penalty_")]:
            axes[3].plot(du, [r[s][k] for r in dl], ls, lw=0.8, alpha=0.6, label=k.replace("swap_penalty_", f"{s.split('_')[0]}: "))
    axes[2].set_title("dev reference-answer NLL", fontsize=9)
    axes[3].set_title("dev image-swap penalty ΔNLL (partner − correct)", fontsize=9)
    axes[3].axhline(0, c="k", lw=0.6)
    axes[2].legend(fontsize=6)
    axes[3].legend(fontsize=6)
    for ax in axes:
        for b in bounds:
            ax.axvline(b, c="gray", ls=":", lw=0.8)
        ax.set_xlabel("update")
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=200)


def fmt(ci, scale=100):
    return f"{scale * ci['mean']:.1f} [{scale * ci['ci_low']:.1f}, {scale * ci['ci_high']:.1f}]"


def tables(run, out):
    s = load_json(results_root() / "task2" / run / "final_eval/summary.json")
    rows, types, advs = [], [], []
    for suite, name in SUITES.items():
        for c in CONDS:
            r = s[suite]["conditions"][c]
            rows.append({"suite": name, "condition": c, "accuracy % [95% CI]": fmt(r["acc"]),
                         "pair acc %": fmt(r["pair_acc"]), "identical %": 100 * r["identical"]["mean"],
                         "empty %": 100 * r["empty"]["mean"], "ref NLL": fmt(r["nll"], 1),
                         "partner-follow %": 100 * r["follow"]["mean"], "ms/answer": 1000 * r["time_per_answer_s"]})
            for qt, v in r["by_type"].items():
                types.append({"suite": name, "condition": c, "type": qt, "acc %": 100 * v["acc"], "ref NLL": v["nll"],
                              "n": v["n_questions"]})
        for c, a in s[suite]["correct_image_advantage"].items():
            advs.append({"suite": name, "control": c, "Δacc (pp) [95% CI]": fmt(a["acc_advantage"]),
                         "ΔNLL [95% CI]": fmt(a["nll_advantage"], 1),
                         **{f"Δacc {qt} (pp)": 100 * v["acc_advantage"] for qt, v in a["by_type"].items()}})
    df = pd.DataFrame(rows)
    (out / "final_table.md").write_text(df.to_markdown(index=False, floatfmt=".1f"), encoding="utf-8")
    (out / "final_table.tex").write_text(df.to_latex(index=False, float_format="%.1f"), encoding="utf-8")
    t = pd.DataFrame(types).pivot_table(index=["suite", "type"], columns="condition", values="acc %")[CONDS]
    (out / "by_type.md").write_text(t.to_markdown(floatfmt=".1f"), encoding="utf-8")
    (out / "advantages.md").write_text(pd.DataFrame(advs).to_markdown(index=False, floatfmt=".1f"), encoding="utf-8")
    print(df.to_markdown(index=False, floatfmt=".1f"))
    print(t.round(1).to_string())


def resources(run, out):
    sm = load_json(results_root() / "task2" / run / "summary.json")
    fc = results_root() / "task2/feature_cache.json"
    ev = load_json(results_root() / "task2" / run / "final_eval/summary.json")
    rows = [{"phase": f"train: {k}", "time_min": v / 60, "peak_GB": sm["stage_peak_mem"].get(k),
             "trainable_params": sm["trainable_params"].get(k)} for k, v in sm["stage_time_s"].items()]
    if fc.exists():
        f = load_json(fc)
        rows.insert(0, {"phase": "feature caching (SigLIP, separate)",
                        "time_min": sum(f[k]["time_s"] for k in f if k.endswith("_cache")) / 60,
                        "peak_GB": f.get("peak_mem_gb"), "trainable_params": 0})
    gen = np.mean([ev[s]["conditions"][c]["time_per_answer_s"] for s in SUITES for c in CONDS])
    rows.append({"phase": "final eval generation (mean ms/answer)", "time_min": None, "peak_GB": ev.get("peak_mem_gb"),
                 "trainable_params": None, "ms_per_answer": 1000 * gen})
    (out / "resources.md").write_text(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".2f"), encoding="utf-8")


def qualitative_fig(run, out, cfg):
    from PIL import Image

    from task2.cache_features import img_path
    from task2.prepare_data import coco_root

    q = load_json(results_root() / "task2" / run / "final_eval/qualitative_pairs.json")["examples"]
    root = coco_root(cfg)
    if not all(img_path(root, "val2014", sl["image_id"]).exists() for e in q for sl in e["slots"]):
        print("COCO val images not available locally; skipping qualitative figure")
        return
    sel = q[:6]
    fig, axes = plt.subplots(len(sel), 2, figsize=(8, 3.1 * len(sel)))
    for r, e in enumerate(sel):
        for s, sl in enumerate(e["slots"]):
            ax = axes[r, s]
            ax.imshow(Image.open(img_path(root, "val2014", sl["image_id"])).convert("RGB"))
            ax.axis("off")
            txt = (f"Q: {e['question']}\nref: {sl['reference']} | correct: {sl['pred_correct']!r}\n"
                   f"swap: {sl['pred_complementary']!r} | none: {sl['pred_none']!r}")
            ax.set_title(textwrap.fill(txt, 60), fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "qualitative.png", dpi=150)


def compression(run, other, out):
    a = load_json(results_root() / "task2" / run / "final_eval/summary.json")
    b = load_json(results_root() / "task2" / other / "final_eval/summary.json")
    sa = load_json(results_root() / "task2" / run / "summary.json")
    sb = load_json(results_root() / "task2" / other / "summary.json")
    rows = []
    for name, s, sm in ((run, a, sa), (other, b, sb)):
        r = {"run": name, "train_min": sm["total_train_time_s"] / 60, "peak_GB": max(sm["stage_peak_mem"].values())}
        for suite in SUITES:
            r[f"{suite} acc (correct)"] = 100 * s[suite]["conditions"]["correct"]["acc"]["mean"]
            r[f"{suite} Δacc vs swap"] = 100 * s[suite]["correct_image_advantage"]["complementary"]["acc_advantage"]["mean"]
            r[f"{suite} ms/answer"] = 1000 * s[suite]["conditions"]["correct"]["time_per_answer_s"]
        rows.append(r)
    (out / "compression.md").write_text(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".1f"), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="vis196")
    ap.add_argument("--compare", default=None)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs/task2.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = results_root() / "task2" / args.run / "report"
    out.mkdir(parents=True, exist_ok=True)
    curves(args.run, out, cfg)
    tables(args.run, out)
    resources(args.run, out)
    qualitative_fig(args.run, out, cfg)
    if args.compare:
        compression(args.run, args.compare, out)


if __name__ == "__main__":
    main()
