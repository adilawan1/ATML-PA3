"""Steps 1-2: repeated sampling and selection on the supplied pools.

For each question, order (10 seeded permutations) and budget N in {1,2,4,8,16,32}: oracle coverage, number
of distinct answer groups, and each selector's answer, correctness and generated-token cost. Outcomes are
averaged over orders within a question, then across questions (per split). Paired question-level
bootstrap (1,000 resamples) for the main accuracy differences.

  python -m task3.offline --dataset math
Outputs: results/task3/<ds>/offline_rows.csv.gz (per question/order/N/method), offline_summary.json,
         offline_question_level.csv.gz, failure_decomposition.json, bootstrap.json
"""
import argparse

import numpy as np
import pandas as pd

from common.utils import REPO_ROOT, load_config, load_json, results_root, save_json
from task3.data import load_pool
from task3.selection import LLM_METHODS, REQUIRED, VLM_METHODS, candidate_scores, select

CFG = load_config(REPO_ROOT / "configs/task3.yaml")


def run(ds):
    kind = CFG["datasets"][ds]["kind"]
    methods = VLM_METHODS if kind == "vlm" else LLM_METHODS
    qids, pool, orders = load_pool(ds)
    rows, cov = [], []
    for qi, q in enumerate(qids):
        p = pool[q]
        for o in range(orders.shape[1]):
            perm = orders[qi, o]
            for N in CFG["offline"]["budgets"]:
                cand = perm[:N]
                scores = candidate_scores(p, cand, kind) if N > 1 else None
                cov.append({"qid": q, "order": o, "N": N, "oracle": bool(p["correct"][cand].any()),
                            "n_groups": len(np.unique(p["group"][cand])), "frac_correct": float(p["correct"][cand].mean()),
                            "cost_all": int(p["ntok"][cand].sum())})
                for m in methods:
                    gsel = select(p, cand, m, scores, kind)
                    cost = int(p["ntok"][cand[0]]) if m == "first" else int(p["ntok"][cand].sum())
                    rows.append({"qid": q, "order": o, "N": N, "method": m, "group": gsel,
                                 "answer": p["reps"][gsel], "correct": bool(p["group_correct"][gsel]), "cost": cost})
        if qi % 100 == 0:
            print(f"{ds}: {qi}/{len(qids)}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(cov), methods


def paired_bootstrap(a, b, n, seed):
    """a, b: question-level accuracies (aligned). Returns mean diff and 95% CI of (a - b)."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a) - np.asarray(b)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    bs = d[idx].mean(axis=1)
    return {"diff": float(d.mean()), "ci_low": float(np.percentile(bs, 2.5)), "ci_high": float(np.percentile(bs, 97.5))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["math", "mathvista"])
    args = ap.parse_args()
    ds = args.dataset
    rd = results_root() / "task3" / ds
    splits = load_json(rd / "splits.json")
    df, cov, methods = run(ds)
    df.to_csv(rd / "offline_rows.csv.gz", index=False)
    cov.to_csv(rd / "offline_coverage_rows.csv.gz", index=False)

    # question level: average over orders
    ql = df.groupby(["qid", "N", "method"]).agg(acc=("correct", "mean"), cost=("cost", "mean")).reset_index()
    qc = cov.groupby(["qid", "N"]).agg(oracle=("oracle", "mean"), n_groups=("n_groups", "mean"),
                                       frac_correct=("frac_correct", "mean")).reset_index()
    ql.to_csv(rd / "offline_question_level.csv.gz", index=False)
    summary, fail, boot = {}, {}, {}
    for split in ("dev", "eval"):
        ids = set(splits[split])
        s = ql[ql.qid.isin(ids)].groupby(["N", "method"]).agg(acc=("acc", "mean"), cost=("cost", "mean")).reset_index()
        c = qc[qc.qid.isin(ids)].groupby("N").agg(oracle=("oracle", "mean"), n_groups=("n_groups", "mean"),
                                                  frac_correct=("frac_correct", "mean")).reset_index()
        summary[split] = {"methods": s.to_dict(orient="records"), "coverage": c.to_dict(orient="records"),
                          "n_questions": len(ids)}
        # failure decomposition per (order, question): no correct candidate vs correct available but missed
        d = df[df.qid.isin(ids)].merge(cov[["qid", "order", "N", "oracle"]], on=["qid", "order", "N"])
        d["sampling_fail"] = ~d.oracle
        d["selection_fail"] = d.oracle & ~d.correct
        f = d.groupby(["qid", "N", "method"])[["correct", "sampling_fail", "selection_fail"]].mean().reset_index()
        fail[split] = f.groupby(["N", "method"])[["correct", "sampling_fail", "selection_fail"]].mean().reset_index().to_dict(orient="records")
        # paired bootstrap of main differences (question-level, after averaging over orders)
        piv = ql[ql.qid.isin(ids)].pivot_table(index=["qid", "N"], columns="method", values="acc").reset_index()
        comps = [("sc", "first"), ("max_K", "max_L"), ("rwc_K", "max_K"), ("rwc_K", "sc"), ("max_K", "sc"), ("shortest", "sc")]
        if "max_VisConf" in methods:
            comps += [("max_VisConf", "max_K"), ("rwc_VisConf", "rwc_K"), ("rwc_VisConf", "sc"), ("max_VisConf", "max_V"),
                      ("rwc_VisConf", "max_VisConf"), ("rwc_V", "sc"), ("max_Ztok", "max_VisConf"), ("rwc_Ztok", "rwc_VisConf")]
        boot[split] = []
        for N, g in piv.groupby("N"):
            for a, b in comps:
                r = paired_bootstrap(g[a], g[b], CFG["offline"]["bootstrap"], CFG["seed"])
                boot[split].append({"N": int(N), "a": a, "b": b, **r})
    save_json({"methods_required": REQUIRED[CFG["datasets"][ds]["kind"]], **summary}, rd / "offline_summary.json")
    save_json(fail, rd / "failure_decomposition.json")
    save_json(boot, rd / "bootstrap.json")
    ev = pd.DataFrame(summary["eval"]["methods"]).pivot(index="method", columns="N", values="acc")
    print((100 * ev).round(1).to_string())
    print(pd.DataFrame(summary["eval"]["coverage"]).round(3).to_string())


if __name__ == "__main__":
    main()
