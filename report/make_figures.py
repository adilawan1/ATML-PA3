"""Report-only figures, redrawn from saved results with larger fonts and legends outside the axes.

Run from the repository root:
    python report/make_figures.py
Writes
  results/task3/report/scaling_report.png    accuracy vs. N (main text, Fig. 3)
  results/task3/report/cost_report.png       accuracy vs. generated tokens (appendix)
  results/task3/report/failures_report.png   sampling vs. selection failures (appendix)
  results/task2/vis196/report/curves_report.png  Task 2 learning curves (main text, Fig. 2)
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

R = Path(__file__).resolve().parents[1] / "results"
T3 = R / "task3"
T2 = R / "task2" / "vis196"
NS = [1, 2, 4, 8, 16, 32]
DATASETS = [("math", "MATH (Qwen2.5-3B)"), ("mathvista", "MathVista (Qwen2.5-VL-3B)")]
LABELS = {
    "first": "FirstAns", "sc": "Self-consistency", "max_L": "Max log-prob",
    "max_K": "Max self-certainty", "rwc_K": "Self-certainty + RWC",
    "max_V": "Max vision", "rwc_V": "Vision + RWC",
    "max_VisConf": "Max VisConf", "rwc_VisConf": "VisConf + RWC",
}
COLORS = {
    "first": "#1f77b4", "sc": "#ff7f0e", "max_L": "#2ca02c", "max_K": "#d62728",
    "rwc_K": "#9467bd", "max_V": "#7f7f7f", "rwc_V": "#bcbd22",
    "max_VisConf": "#8c564b", "rwc_VisConf": "#e377c2",
}


def task3_curves(xkey, xlabel, out):
    plt.rcParams.update({"font.size": 12})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (ds, title) in zip(axes, DATASETS):
        ev = json.loads((T3 / ds / "offline_summary.json").read_text())["eval"]
        rows = {(r["method"], r["N"]): r for r in ev["methods"]}
        cov = {r["N"]: 100 * r["oracle"] for r in ev["coverage"]}
        if xkey == "N":
            ax.plot(NS, [cov[n] for n in NS], "k--", lw=2, label="Oracle coverage")
        else:  # oracle coverage consumes every candidate, as SC does
            ax.plot([rows[("sc", n)]["cost"] for n in NS], [cov[n] for n in NS], "k--", lw=2, label="Oracle coverage")
        for m in LABELS:
            if (m, 1) not in rows:
                continue
            xs = NS if xkey == "N" else [rows[(m, n)]["cost"] for n in NS]
            ax.plot(xs, [100 * rows[(m, n)]["acc"] for n in NS], marker="o", ms=4, lw=1.8,
                    color=COLORS[m], label=LABELS[m])
        if xkey == "N":
            ax.set_xscale("log", base=2)
            ax.set_xticks(NS, [str(n) for n in NS])
        else:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("held-out accuracy (%)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=10.5, frameon=False)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


def task3_failures(out):
    plt.rcParams.update({"font.size": 12})
    groups = {"math": ["sc", "rwc_K"], "mathvista": ["sc", "rwc_K", "rwc_VisConf"]}
    parts = [("correct", "correct", 1.0), ("selection_fail", "correct candidate missed", 0.55),
             ("sampling_fail", "no correct candidate", 0.2)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (ds, title) in zip(axes, DATASETS):
        rows = {(r["method"], r["N"]): r for r in json.loads((T3 / ds / "failure_decomposition.json").read_text())["eval"]}
        ms = groups[ds]
        width = 0.8 / len(ms)
        x = np.arange(len(NS))
        for j, m in enumerate(ms):
            bottom = np.zeros(len(NS))
            for key, _, alpha in parts:
                vals = np.array([100 * rows[(m, n)][key] for n in NS])
                ax.bar(x + (j - (len(ms) - 1) / 2) * width, vals, width * 0.95, bottom=bottom,
                       color=COLORS[m], alpha=alpha, edgecolor="white", linewidth=0.5)
                bottom += vals
        ax.set_xticks(x, [str(n) for n in NS])
        ax.set_xlabel("candidates N")
        ax.set_ylabel("% of question-orders")
        ax.set_ylim(0, 100)
        ax.set_title(title)
    method_handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[m]) for m in ["sc", "rwc_K", "rwc_VisConf"]]
    part_handles = [plt.Rectangle((0, 0), 1, 1, color="#555555", alpha=a) for _, _, a in parts]
    fig.legend(method_handles + part_handles,
               [LABELS[m] for m in ["sc", "rwc_K", "rwc_VisConf"]] + [p[1] for p in parts],
               loc="lower center", ncol=3, fontsize=11, frameon=False)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


def task2_curves(out):
    plt.rcParams.update({"font.size": 14, "axes.titlesize": 14})
    train = [json.loads(line) for line in (T2 / "train_log.jsonl").read_text().splitlines() if line.strip()]
    dev = [json.loads(line) for line in (T2 / "dev_log.jsonl").read_text().splitlines() if line.strip()]
    du = [d["update"] for d in dev]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.9))

    u = np.array([t["update"] for t in train])
    loss = np.array([t["loss"] for t in train])
    k = 50
    ma = np.convolve(loss, np.ones(k) / k, mode="valid")
    axes[0].plot(u, loss, color="#9ecae1", lw=0.4)
    axes[0].plot(u[k - 1:], ma, color="#08519c", lw=1.6)
    axes[0].set_title("training loss")

    axes[1].plot(du, [d["caption_loss"] for d in dev], "o-", color="#08519c", ms=4)
    axes[1].set_title("dev caption loss")

    suites = [("grounding_dev", "#1f77b4"), ("vqa_dev", "#ff7f0e")]
    styles = [("nll_correct", "-"), ("nll_complementary", "--"), ("nll_none", ":")]
    for s, c in suites:
        for key, ls in styles:
            axes[2].plot(du, [d[s][key] for d in dev], ls, color=c, lw=1.8, marker="o", ms=3)
    axes[2].set_title("dev answer NLL")

    for s, c in suites:
        axes[3].plot(du, [d[s]["swap_penalty"] for d in dev], "-", color=c, lw=2.4, marker="o", ms=4)
    thin = [("grounding_dev", "swap_penalty_presence", "#6baed6"), ("grounding_dev", "swap_penalty_counting", "#08306b"),
            ("vqa_dev", "swap_penalty_yes/no", "#fdae6b"), ("vqa_dev", "swap_penalty_number", "#e6550d"),
            ("vqa_dev", "swap_penalty_other", "#7f2704")]
    for s, key, c in thin:
        axes[3].plot(du, [d[s][key] for d in dev], "-", color=c, lw=0.9, alpha=0.9)
    axes[3].axhline(0, color="k", lw=0.6)
    axes[3].set_title(r"dev swap penalty $\Delta_{NLL}$")

    for ax in axes:
        for b in (2250, 3750):
            ax.axvline(b, color="grey", ls=":", lw=1)
        ax.set_xlabel("update")
        ax.grid(alpha=0.25)

    legend = [
        Line2D([], [], color="#1f77b4", lw=2, label="generated suite"),
        Line2D([], [], color="#ff7f0e", lw=2, label="VQAv2 suite"),
        Line2D([], [], color="k", ls="-", label="correct image"),
        Line2D([], [], color="k", ls="--", label="complementary"),
        Line2D([], [], color="k", ls=":", label="no image"),
        Line2D([], [], color="#6baed6", lw=1, label="presence"),
        Line2D([], [], color="#08306b", lw=1, label="counting"),
        Line2D([], [], color="#fdae6b", lw=1, label="yes/no"),
        Line2D([], [], color="#e6550d", lw=1, label="number"),
        Line2D([], [], color="#7f2704", lw=1, label="other"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=10, fontsize=11.5, frameon=False,
               handlelength=1.4, columnspacing=0.8, handletextpad=0.4)
    for ax in axes:
        ax.set_xticks([0, 2000, 4000])
    fig.tight_layout(rect=(0, 0.09, 1, 1), w_pad=0.6)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    task3_curves("N", "candidates N", T3 / "report" / "scaling_report.png")
    task3_curves("cost", "mean generated tokens per question", T3 / "report" / "cost_report.png")
    task3_failures(T3 / "report" / "failures_report.png")
    task2_curves(T2 / "report" / "curves_report.png")
