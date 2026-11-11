"""Extra statistics cited in the report. Run from the repository root:
    python report/extra_stats.py

Writes
  results/task1/report/paired_tests.csv   paired differences + McNemar tests (Task 1)
  results/task2/vis196/report/extra_stats.json   swap outcome split, yes/no constant baselines
  results/task3/report/extra_deltas.json  argmax deltas vs SC@32 not in bootstrap.json
"""
import csv
import gzip
import json
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1] / "results"

# ---------------- Task 1: paired tests on identical items ----------------
METRIC = {"gsm8k": "exact_match", "hellaswag": "acc_norm", "boolq": "acc", "arc_easy": "acc_norm"}
CONDS = ["baseline", "full_ft_mid", "full_ft", "full_ft_l2sp", "lora4", "lora16", "oplora16"]


def load(cond, bench):
    with gzip.open(R / "task1/eval" / cond / f"{bench}_samples.jsonl.gz", "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    if bench == "gsm8k":
        rows = [r for r in rows if r["filter"] == "flexible-extract"]
    return {r["doc_id"]: float(r[METRIC[bench]]) for r in rows}


S = {(c, b): load(c, b) for c in CONDS for b in METRIC}


def paired(a, b, bench):
    A, B = S[(a, bench)], S[(b, bench)]
    ids = sorted(A)
    x = np.array([A[i] for i in ids])
    y = np.array([B[i] for i in ids])
    d = x - y
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    n10, n01 = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
    n = n10 + n01
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(n10, n01) + 1)) / 2**n) if n else 1.0
    return dict(a=a, b=b, benchmark=bench, n=len(d), diff_pp=100 * m, ci_low_pp=100 * (m - 1.96 * se),
                ci_high_pp=100 * (m + 1.96 * se), a_only=n10, b_only=n01, mcnemar_p=p)


pairs = [(c, "baseline") for c in CONDS[1:]] + [
    ("full_ft", "full_ft_mid"), ("lora16", "lora4"), ("full_ft_l2sp", "full_ft"),
    ("oplora16", "lora16"), ("lora16", "full_ft")]
out = [paired(a, b, bm) for a, b in pairs for bm in METRIC]
with open(R / "task1/report/paired_tests.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)

# ---------------- Task 2: swap outcome split, yes/no constant answers ----------------


def rows2(name):
    with open(R / "task2/vis196/final_eval" / name, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def soft(a, ans):
    return float(np.mean([min(1, sum(ans[j] == a for j in range(10) if j != i) / 3) for i in range(10)]))


t2 = {}
for suite, fname, score in [("generated", "grounding_test_rows.jsonl", "em"), ("vqav2", "vqa_test_rows.jsonl", "acc")]:
    sw = [r for r in rows2(fname) if r["condition"] == "complementary"]
    orig = float(np.mean([float(r[score]) for r in sw]))
    partner = float(np.mean([float(r["partner_follow"]) for r in sw]))
    t2[f"{suite}_swap"] = dict(original_ref=orig, partner_ref=partner, neither=1 - orig - partner)
yn = [r for r in rows2("vqa_test_rows.jsonl") if r["condition"] == "correct" and r["qtype"] == "yes/no"]
t2["vqav2_yesno"] = {
    "n_questions": len(yn),
    "model_correct_image": float(np.mean([float(r["acc"]) for r in yn])),
    **{f"constant_{c}": float(np.mean([soft(c, [s.strip().lower() for s in r["answers"]]) for r in yn]))
       for c in ("yes", "no")},
}
(R / "task2/vis196/report/extra_stats.json").write_text(json.dumps(t2, indent=2))

# ---------------- Task 3: remaining argmax deltas vs SC@32 ----------------
t3 = {}
for ds, methods in [("math", ["max_L"]),
                    ("mathvista", ["max_L", "max_V", "max_VisConf", "max_Ztok", "rwc_Ztok"])]:
    q = pd.read_csv(R / "task3" / ds / "offline_question_level.csv.gz")
    ev = set(map(str, json.loads((R / "task3" / ds / "splits.json").read_text())["eval"]))
    q["qid"] = q["qid"].astype(str)
    P = q[(q.N == 32) & q.qid.isin(ev)].pivot(index="qid", columns="method", values="acc")
    rng = np.random.default_rng(6304)
    for m in methods:
        d = (P[m] - P["sc"]).to_numpy()
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        t3[f"{ds}/{m}"] = dict(n=len(d), diff_pp=100 * d.mean(), ci_low_pp=100 * lo, ci_high_pp=100 * hi)
(R / "task3/report/extra_deltas.json").write_text(json.dumps(t3, indent=2))

print(json.dumps(t2, indent=1))
print(json.dumps(t3, indent=1))
