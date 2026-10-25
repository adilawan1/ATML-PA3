"""Step 3: what the quality signals capture (N = 32, full pool; gold labels used only for analysis).

- Within-question associations of rollout length and each score with correctness (Spearman / point-biserial),
  averaged over questions with mixed correctness.
- Top-vs-bottom quartile correctness gap per score within a question, averaged across questions
  (all questions, and mixed-correctness questions only). Shortest rollout = length-only control.
- MathVista: per-rollout correlation between calibrated confidence c_t and visual engagement v_t,
  summarized across questions; check Z_token - Z = Cov_t(c, v); fixed illustrative trajectories.

  python -m task3.signals --dataset mathvista
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from common.utils import REPO_ROOT, load_config, load_json, results_root, save_json
from task3.data import load_pool
from task3.selection import visconf_scores

CFG = load_config(REPO_ROOT / "configs/task3.yaml")


def score_table(p, kind):
    n = len(p["group"])
    cand = np.arange(n)
    s = {"L": p["L"], "K": p["K"], "neg_length": -p["ntok"].astype(float), "length": p["ntok"].astype(float)}
    if kind == "vlm":
        C, Z, Zt = visconf_scores(p, cand)
        s.update({"V": p["V"], "C": C, "VisConf": Z, "Ztok": Zt})
    return s


def quartile_gap(score, correct):
    """Mean correctness of the top quarter minus the bottom quarter by score (ties by rollout position)."""
    n = len(score)
    k = n // 4
    order = np.argsort(-np.where(np.isfinite(score), score, -np.inf), kind="stable")
    return correct[order[:k]].mean() - correct[order[-k:]].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["math", "mathvista"])
    ap.add_argument("--split", default="eval")
    ap.add_argument("--n_examples", type=int, default=3)
    args = ap.parse_args()
    ds, kind = args.dataset, CFG["datasets"][args.dataset]["kind"]
    qids, pool, orders = load_pool(ds)
    splits = load_json(results_root() / "task3" / ds / "splits.json")
    ids = splits[args.split]

    per_score = {}
    covcheck, cv_corr_q, cv_corr_by_correct = [], [], {True: [], False: []}
    len_score_corr = {}
    for q in ids:
        p = pool[q]
        corr = p["correct"].astype(float)
        mixed = 0 < corr.mean() < 1
        s = score_table(p, kind)
        for name, v in s.items():
            d = per_score.setdefault(name, {"gap_all": [], "gap_mixed": [], "rho_mixed": []})
            g = quartile_gap(v, corr)
            d["gap_all"].append(g)
            if mixed:
                d["gap_mixed"].append(g)
                fin = np.isfinite(v)
                if fin.sum() > 2 and np.std(v[fin]) > 0:
                    d["rho_mixed"].append(spearmanr(v[fin], corr[fin]).statistic)
            if name not in ("length", "neg_length"):
                fin = np.isfinite(v)
                if fin.sum() > 2 and np.std(v[fin]) > 0:
                    len_score_corr.setdefault(name, []).append(spearmanr(v[fin], p["ntok"][fin]).statistic)
        if kind == "vlm":
            cand = np.arange(len(p["group"]))
            toks = np.concatenate([p["kl"][i] for i in cand])
            q05, q95 = np.percentile(toks, [5, 95])
            rs = []
            for i in cand:
                if len(p["kl"][i]) < 3:
                    continue
                c = np.clip((p["kl"][i] - q05) / max(q95 - q05, 1e-8), 0, 1)
                v = p["v"][i]
                ok = np.isfinite(v)
                if ok.sum() > 2 and c[ok].std() > 0 and v[ok].std() > 0:
                    r = np.corrcoef(c[ok], v[ok])[0, 1]
                    rs.append(r)
                    cv_corr_by_correct[bool(p["correct"][i])].append(r)
                    # identity check: mean(c v) - mean(c) mean(v) = population covariance
                    covcheck.append(abs((np.mean(c[ok] * v[ok]) - c[ok].mean() * v[ok].mean())
                                        - np.cov(c[ok], v[ok], bias=True)[0, 1]))
            if rs:
                cv_corr_q.append(float(np.mean(rs)))

    def stats(x):
        x = np.asarray([v for v in x if np.isfinite(v)])
        return {"mean": float(x.mean()) if len(x) else None, "median": float(np.median(x)) if len(x) else None, "n": len(x)}

    out = {"split": args.split, "n_questions": len(ids),
           "scores": {k: {"quartile_gap_all": stats(v["gap_all"]), "quartile_gap_mixed": stats(v["gap_mixed"]),
                          "spearman_with_correct_mixed": stats(v["rho_mixed"])} for k, v in per_score.items()},
           "spearman_score_vs_length": {k: stats(v) for k, v in len_score_corr.items()},
           "n_mixed_questions": int(sum(0 < pool[q]["correct"].mean() < 1 for q in ids))}
    if kind == "vlm":
        out["confidence_vision_corr"] = {"per_question_mean": stats(cv_corr_q),
                                         "rollouts_correct": stats(cv_corr_by_correct[True]),
                                         "rollouts_incorrect": stats(cv_corr_by_correct[False]),
                                         "max_abs_cov_identity_error": float(max(covcheck)) if covcheck else None}
        illustrate(pool, ids, ds, args.n_examples, orders, qids)
    save_json(out, results_root() / "task3" / ds / "signals.json")
    for k, v in out["scores"].items():
        print(f"{k:12s} gap_all {v['quartile_gap_all']['mean']:.3f}  gap_mixed {v['quartile_gap_mixed']['mean']}  "
              f"rho_mixed {v['spearman_with_correct_mixed']['mean']}")
    if kind == "vlm":
        print("c-v corr:", out["confidence_vision_corr"])


def illustrate(pool, ids, ds, n, orders, qids):
    """Fixed rule: the first n questions of the split (permutation order) where max-VisConf and max-K select
    different answers at N=32; plot c_t and v_t (moving average) for the two selected rollouts."""
    from task3.selection import argmax_earliest

    chosen = []
    for q in ids:
        p = pool[q]
        cand = orders[qids.index(q), 0]
        C, Z, _ = visconf_scores(p, cand)
        iz, ik = cand[argmax_earliest(Z)], cand[argmax_earliest(p["K"][cand])]
        if p["group"][iz] != p["group"][ik]:
            chosen.append((q, iz, ik, C))
        if len(chosen) == n:
            break
    if not chosen:
        return
    fig, axes = plt.subplots(len(chosen), 2, figsize=(11, 2.6 * len(chosen)), squeeze=False)
    rows = []
    for r, (q, iz, ik, _) in enumerate(chosen):
        p = pool[q]
        toks = np.concatenate(p["kl"])
        q05, q95 = np.percentile(toks, [5, 95])
        for c_, (i, lab) in enumerate([(iz, "max VisConf"), (ik, "max self-certainty")]):
            c = np.clip((p["kl"][i] - q05) / max(q95 - q05, 1e-8), 0, 1)
            v = p["v"][i]
            w = max(1, len(c) // 50)
            ma = lambda x: np.convolve(np.nan_to_num(x), np.ones(w) / w, mode="valid")
            ax = axes[r, c_]
            ax.plot(ma(c), label="confidence c_t", lw=1)
            ax2 = ax.twinx()
            ax2.plot(ma(v), color="C1", label="vision v_t", lw=1)
            ax.set_title(f"q={q} {lab}: ans={p['reps'][p['group'][i]]!s:.20} "
                         f"({'correct' if p['correct'][i] else 'wrong'}), T={len(c)}", fontsize=8)
            ax.set_xlabel("generated token (moving avg.)", fontsize=7)
            rows.append({"qid": q, "selector": lab, "rollout": int(p["rollout_ids"][i]),
                         "answer": p["reps"][p["group"][i]], "correct": bool(p["correct"][i]), "T": len(c),
                         "mean_c": float(c.mean()), "mean_v": float(np.nanmean(v))})
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    d = results_root() / "task3" / ds
    fig.savefig(d / "illustrative_trajectories.png", dpi=180)
    save_json({"rule": "first questions in split order where max-VisConf and max-K disagree at N=32 (order 0)",
               "examples": rows}, d / "illustrative_trajectories.json")


if __name__ == "__main__":
    main()
