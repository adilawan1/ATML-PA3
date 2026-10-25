"""Task 3 figures and tables (held-out evaluation split unless stated).

  python -m task3.analyze
Outputs in results/task3/report/: scaling.png, cost.png, failures.png, adaptive.png,
method_table.{md,tex}, failure_table.md, adaptive_table.md, live_table.md, selection_disagreements.json
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.utils import REPO_ROOT, load_config, load_json, results_root, save_json
from task3.selection import LABELS, REQUIRED

CFG = load_config(REPO_ROOT / "configs/task3.yaml")
DS = {"math": "MATH (Qwen2.5-3B)", "mathvista": "MathVista (Qwen2.5-VL-3B)"}
OUT = results_root() / "task3/report"
SHOW = ["first", "sc", "max_L", "max_K", "rwc_K", "max_VisConf", "rwc_VisConf", "max_V", "rwc_V"]


def load(ds, f):
    p = results_root() / "task3" / ds / f
    return load_json(p) if p.exists() else None


def scaling():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, ax2, ds in zip(axes, axes2, DS):
        s = load(ds, "offline_summary.json")
        if s is None:
            continue
        m = pd.DataFrame(s["eval"]["methods"])
        c = pd.DataFrame(s["eval"]["coverage"])
        ax.plot(c.N, 100 * c.oracle, "k--", label="oracle coverage")
        ax2.plot(m[m.method == "sc"].cost, 100 * c.oracle, "k--", label="oracle coverage")
        for meth in SHOW:
            d = m[m.method == meth].sort_values("N")
            if len(d):
                ax.plot(d.N, 100 * d.acc, marker="o", ms=3, label=LABELS[meth])
                ax2.plot(d.cost, 100 * d.acc, marker="o", ms=3, label=LABELS[meth])
        for a, xl in ((ax, "candidates N"), (ax2, "mean generated tokens per question")):
            a.set_xscale("log", base=2 if a is ax else 10)
            a.set_xlabel(xl)
            a.set_ylabel("accuracy (%)")
            a.set_title(DS[ds], fontsize=10)
            a.grid(alpha=0.3)
        ax.set_xticks(CFG["offline"]["budgets"], CFG["offline"]["budgets"])
    axes[-1].legend(fontsize=7, loc="lower right")
    axes2[-1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "scaling.png", dpi=200)
    fig2.tight_layout()
    fig2.savefig(OUT / "cost.png", dpi=200)


def method_table():
    rows = []
    for ds in DS:
        s, b = load(ds, "offline_summary.json"), load(ds, "bootstrap.json")
        if s is None:
            continue
        m = pd.DataFrame(s["eval"]["methods"])
        bt = pd.DataFrame(b["eval"])
        for meth in m.method.unique():
            r = {"dataset": ds, "method": LABELS[meth], "required": meth in s["methods_required"]}
            for N in (4, 8, 32):
                d = m[(m.method == meth) & (m.N == N)]
                r[f"acc@{N}"] = 100 * d.acc.iat[0]
            r["tokens@32"] = m[(m.method == meth) & (m.N == 32)].cost.iat[0]
            x = bt[(bt.N == 32) & (bt.a == meth) & (bt.b == "sc")]
            if len(x):
                r["Δ vs SC @32 (pp) [95% CI]"] = f"{100 * x["diff"].iat[0]:+.1f} [{100 * x.ci_low.iat[0]:+.1f}, {100 * x.ci_high.iat[0]:+.1f}]"
            rows.append(r)
        cov = pd.DataFrame(s["eval"]["coverage"])
        rows.append({"dataset": ds, "method": "Oracle coverage", **{f"acc@{N}": 100 * cov[cov.N == N].oracle.iat[0] for N in (4, 8, 32)}})
    if rows:
        df = pd.DataFrame(rows)
        (OUT / "method_table.md").write_text(df.to_markdown(index=False, floatfmt=".1f"), encoding="utf-8")
        (OUT / "method_table.tex").write_text(df.to_latex(index=False, float_format="%.1f"), encoding="utf-8")
        print(df.to_markdown(index=False, floatfmt=".1f"))


def failures():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    rows = []
    for ax, ds in zip(axes, DS):
        f = load(ds, "failure_decomposition.json")
        if f is None:
            continue
        d = pd.DataFrame(f["eval"])
        sel = d[d.method.isin(["sc", "rwc_K"] + (["rwc_VisConf"] if ds == "mathvista" else []))]
        for i, (meth, g) in enumerate(sel.groupby("method", sort=False)):
            g = g.sort_values("N")
            x = np.arange(len(g)) + i * 0.27
            ax.bar(x, 100 * g.correct, 0.25, label=f"{LABELS[meth]}: correct", color=f"C{i}")
            ax.bar(x, 100 * g.selection_fail, 0.25, bottom=100 * g.correct, color=f"C{i}", alpha=0.4,
                   label=f"{LABELS[meth]}: missed correct")
            ax.bar(x, 100 * g.sampling_fail, 0.25, bottom=100 * (g.correct + g.selection_fail), color="lightgray",
                   label="no correct candidate" if i == 0 else None)
            for _, r in g.iterrows():
                rows.append({"dataset": ds, "method": LABELS[meth], "N": int(r.N), "correct": 100 * r.correct,
                             "selection_fail": 100 * r.selection_fail, "sampling_fail": 100 * r.sampling_fail})
        ax.set_xticks(np.arange(len(CFG["offline"]["budgets"])) + 0.27, CFG["offline"]["budgets"])
        ax.set_xlabel("candidates N")
        ax.set_ylabel("% of question-orders")
        ax.set_title(DS[ds], fontsize=10)
    for ax in axes:
        ax.legend(fontsize=6, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    fig.savefig(OUT / "failures.png", dpi=200, bbox_inches="tight")
    if rows:
        (OUT / "failure_table.md").write_text(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".1f"), encoding="utf-8")


def adaptive():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    rows = []
    for ax, ds in zip(axes, DS):
        a, s = load(ds, "adaptive.json"), load(ds, "offline_summary.json")
        if a is None or s is None:
            continue
        m = pd.DataFrame(s["eval"]["methods"])
        sc = m[m.method == "sc"].sort_values("N")
        ax.plot(sc.cost, 100 * sc.acc, "o-", c="gray", label="fixed-N self-consistency")
        for N, x, y in zip(sc.N, sc.cost, sc.acc):
            ax.annotate(f"N={N}", (x, 100 * y), fontsize=6, textcoords="offset points", xytext=(3, -8))
        for i, fam in enumerate(["agreement", "quality"]):
            h = a["heldout"][fam]
            ax.scatter([h["cost"]], [100 * h["acc"]], s=60, marker="*", c=f"C{i + 1}",
                       label=f"{fam} (τ={h['setting']['tau']}, γ={h['setting']['gamma']})")
            rows.append({"dataset": ds, "policy": fam, "tau": h["setting"]["tau"], "gamma": h["setting"]["gamma"],
                         "acc": 100 * h["acc"], "tokens": h["cost"], "mean_rollouts": h["mean_rollouts"],
                         **{f"stop@{k}": 100 * v for k, v in h["stop_distribution"].items()},
                         "incorrect_early_%": 100 * h["incorrect_early_stops_rate"],
                         "Δacc vs fixed32 (pp)": 100 * h["bootstrap_vs_fixed32"]["diff"]})
        f32 = a["heldout"]["fixed32_sc"]
        rows.append({"dataset": ds, "policy": "fixed-32 SC", "acc": 100 * f32["acc"], "tokens": f32["cost"], "mean_rollouts": 32})
        ax.set_xscale("log")
        ax.set_xlabel("mean generated tokens per question")
        ax.set_ylabel("accuracy (%)")
        ax.set_title(DS[ds], fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "adaptive.png", dpi=200)
    if rows:
        df = pd.DataFrame(rows)
        (OUT / "adaptive_table.md").write_text(df.to_markdown(index=False, floatfmt=".1f"), encoding="utf-8")
        print(df.to_markdown(index=False, floatfmt=".1f"))


def live():
    s = load("live", "summary.json")
    if s is None:
        return
    df = pd.DataFrame(s["strategies"])
    (OUT / "live_table.md").write_text(df.to_markdown(index=False, floatfmt=".3f")
                                       + f"\n\nTransitions (initial→revised): {s['refine_transitions']}\n", encoding="utf-8")
    print(df.to_markdown(index=False, floatfmt=".3f"))


def disagreements(n=5):
    """Fixed rule: first n eval questions (split order) where SC and the best quality selector disagree at
    N=32 on order 0, with each candidate answer's vote count and correctness."""
    out = {}
    for ds in DS:
        p = results_root() / "task3" / ds / "offline_rows.csv.gz"
        if not p.exists():
            continue
        rows = pd.read_csv(p)
        splits = load(ds, "splits.json")
        other = "rwc_VisConf" if ds == "mathvista" else "rwc_K"
        r = rows[(rows.N == 32) & (rows.order == 0) & rows.method.isin(["sc", other])]
        piv = r.pivot(index="qid", columns="method", values=["answer", "correct"])
        groups = load(ds, "answer_groups.json")
        ex = []
        for q in splits["eval"]:
            q2 = int(q) if q.isdigit() and int(q) in piv.index else q
            if q2 not in piv.index:
                continue
            a = piv.loc[q2]
            if a[("answer", "sc")] != a[("answer", other)]:
                g = groups[q]
                ex.append({"qid": q, "sc_answer": a[("answer", "sc")], "sc_correct": bool(a[("correct", "sc")]),
                           f"{other}_answer": a[("answer", other)], f"{other}_correct": bool(a[("correct", other)]),
                           "votes": dict(zip(g["reps"], g["counts"])), "gold": g["gold"]})
            if len(ex) == n:
                break
        out[ds] = ex
    save_json(out, OUT / "selection_disagreements.json")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scaling()
    method_table()
    failures()
    adaptive()
    live()
    disagreements()


if __name__ == "__main__":
    main()
