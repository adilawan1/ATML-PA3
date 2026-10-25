"""Load the supplied rollouts, verify alignment, build answer groups / correctness, splits and orders.

  python -m task3.data --dataset math --inspect     # print schemas (check column names first)
  python -m task3.data --dataset math               # build pool + splits + orders
  python -m task3.data --dataset mathvista

Outputs
  $PA3_OUT/task3/<ds>/pool.pkl           per-question arrays (scores, token measurements, groups)
  results/task3/<ds>/splits.json         dev / eval / live question IDs
  results/task3/<ds>/orders.npz          10 seeded permutations of the 32 rollout positions per question
  results/task3/<ds>/data_checks.json    alignment checks, group statistics, scorer version
"""
import argparse
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from common.utils import REPO_ROOT, load_config, load_json, out_root, results_root, save_json
from task3 import answers as A

CFG_PATH = REPO_ROOT / "configs/task3.yaml"


def data_dir(cfg, ds):
    root = Path(os.environ.get("PA3_T3_DATA", REPO_ROOT / cfg["data_root"]))
    return root / cfg["datasets"][ds]["folder"]


def resolve(df, cfg, field, required=True):
    for c in cfg["columns"][field]:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"none of {cfg['columns'][field]} in columns {list(df.columns)}; edit configs/task3.yaml")
    return None


def inspect(cfg, ds):
    d = data_dir(cfg, ds)
    for f in sorted(d.glob("*.parquet")):
        df = pd.read_parquet(f)
        print(f"\n== {f.name}: {df.shape}")
        print(df.dtypes)
        row = df.iloc[0].to_dict()
        print({k: (v[:5] if hasattr(v, "__len__") and not isinstance(v, str) else (v[:120] if isinstance(v, str) else v))
               for k, v in row.items()})
    for f in list(d.glob("*.md")) + list(d.glob("*.MD")) + list(d.parent.glob("README*")):
        print(f"\n== {f}")
        print(f.read_text(encoding="utf-8")[:6000])


def load_joined(cfg, ds):
    d = data_dir(cfg, ds)
    gen = pd.read_parquet(d / "generations.parquet")
    prob = pd.read_parquet(d / "probability_metrics.parquet")
    keys = [resolve(gen, cfg, k) for k in ("dataset", "sample_id", "rollout_id")]
    pkeys = [resolve(prob, cfg, k) for k in ("dataset", "sample_id", "rollout_id")]
    prob = prob.rename(columns=dict(zip(pkeys, keys)))
    dup = set(gen.columns) & set(prob.columns) - set(keys)
    df = gen.merge(prob.drop(columns=list(dup)), on=keys, how="inner", validate="one_to_one")
    if cfg["datasets"][ds]["kind"] == "vlm":
        att = pd.read_parquet(d / "attention_metrics.parquet")
        akeys = [resolve(att, cfg, k) for k in ("dataset", "sample_id", "rollout_id")]
        att = att.rename(columns=dict(zip(akeys, keys)))
        dup = set(df.columns) & set(att.columns) - set(keys)
        df = df.merge(att.drop(columns=list(dup)), on=keys, how="inner", validate="one_to_one")
    assert len(df) == len(gen), f"join lost rows: {len(gen)} -> {len(df)}"
    return df, keys


def make_splits(qids, cfg):
    ids = sorted(str(q) for q in qids)  # lexicographic
    perm = np.random.default_rng(cfg["seed"]).permutation(np.array(ids, dtype=object)).tolist()
    n_dev = cfg["split"]["n_dev"]
    if len(ids) != 1000:  # smoke tests on small synthetic pools only; real pools have exactly 1,000 questions
        n_dev = max(1, round(0.2 * len(ids)))
    return {"permuted": perm, "dev": perm[:n_dev], "eval": perm[n_dev:], "live": perm[n_dev:n_dev + cfg["split"]["n_live"]]}


def make_orders(qids_sorted, n_roll, cfg):
    rng = np.random.default_rng(cfg["seed"])
    return np.stack([np.stack([rng.permutation(n_roll) for _ in range(cfg["offline"]["n_orders"])]) for _ in qids_sorted])


def _group_math_worker(args):
    answers_by_q = args
    mv = A.MathVerifier()
    out = []
    for answers in answers_by_q:
        out.append(A.group_math(answers, mv))
    return out


def _grade_math_worker(args):
    mv = A.MathVerifier()
    return [A.grade_groups(reps, gold, "math_verify", mv=mv) for reps, gold in args]


def chunks(xs, n):
    k = max(1, len(xs) // n + 1)
    return [xs[i:i + k] for i in range(0, len(xs), k)]


def to_arr(x):
    return np.asarray(x, dtype=np.float64) if x is not None else np.zeros(0)


def build(cfg, ds, workers):
    dcfg = cfg["datasets"][ds]
    df, (c_ds, c_sid, c_rid) = load_joined(cfg, ds)
    col = {f: resolve(df, cfg, f, required=False) for f in cfg["columns"]}
    for f in ("extracted", "gold", "logp", "kl"):
        assert col[f], f"missing required column for {f}"
    df["_sid"] = df[c_sid].astype(str)
    df = df.sort_values(["_sid", c_rid]).reset_index(drop=True)
    qids = sorted(df["_sid"].unique())
    checks = {"n_rows": len(df), "n_questions": len(qids), "columns_used": col}

    # ---- alignment checks
    n_lp = df[col["logp"]].map(len)
    n_kl = df[col["kl"]].map(len)
    checks["logp_vs_kl_len_mismatch"] = int((n_lp != n_kl).sum())
    if col["token_ids"]:
        diff = df[col["token_ids"]].map(len) - n_lp
        checks["token_ids_minus_logp_len_counts"] = {str(k): int(v) for k, v in diff.value_counts().items()}
    if col["n_tokens"]:
        diff = df[col["n_tokens"]] - n_lp
        checks["n_tokens_minus_logp_len_counts"] = {str(k): int(v) for k, v in diff.value_counts().items()}
    if dcfg["kind"] == "vlm":
        for f in ("attn_image", "attn_prompt", "attn_generated"):
            assert col[f], f"missing attention column {f}"
            checks[f"{f}_len_mismatch"] = int((df[col[f]].map(len) != n_lp).sum())
    # generated-token cost: supplied retained count if present, else token-measurement length
    cost = df[col["n_tokens"]].to_numpy() if col["n_tokens"] else n_lp.to_numpy()
    checks["cost_source"] = col["n_tokens"] or "len(logp)"

    # ---- answer groups (no gold) then grading (gold)
    per_q = {q: g for q, g in df.groupby("_sid", sort=True)}
    answers = [per_q[q][col["extracted"]].tolist() for q in qids]
    golds = [per_q[q][col["gold"]].iloc[0] for q in qids]
    meta = None
    if dcfg["grader"] == "math_verify":
        with ProcessPoolExecutor(workers) as ex:
            grouped = [g for part in ex.map(_group_math_worker, chunks(answers, workers)) for g in part]
        with ProcessPoolExecutor(workers) as ex:
            gcorrect = [g for part in ex.map(_grade_math_worker, chunks([(r, gd) for (_, r), gd in zip(grouped, golds)], workers))
                        for g in part]
        checks["scorer"] = f"math-verify {version('math-verify')}"
    else:
        meta_file = os.environ.get("PA3_T3_META")  # optional local JSON {pid: record} (tests / offline use)
        if meta_file:
            meta = {str(k): v for k, v in load_json(meta_file).items()}
        else:
            from datasets import load_dataset

            md = load_dataset(dcfg["metadata"], split=dcfg["metadata_split"])
            keep = ["pid", "question", "choices", "answer", "question_type", "answer_type", "precision", "unit"]
            md = md.remove_columns([c for c in md.column_names if c not in keep])
            meta = {str(r["pid"]): r for r in md}
            save_json(meta, results_root() / "task3" / ds / "mathvista_metadata.json")
        missing = [q for q in qids if q not in meta]
        assert not missing, f"{len(missing)} sample_ids not found in MathVista metadata (e.g. {missing[:3]})"
        grouped = [A.group_mathvista(a, meta[q]) for a, q in zip(answers, qids)]
        gcorrect = [A.grade_groups(reps, None, "mathvista", meta=meta[q]) for (_, reps), q in zip(grouped, qids)]
        checks["scorer"] = "MathVista normalize_extracted_answer (ported), MC as letters, exact match"
        checks["gold_vs_metadata_mismatch"] = int(sum(
            str(g).strip() != str(meta[q]["answer"]).strip() for g, q in zip(golds, qids)))

    pool = {}
    for qi, q in enumerate(qids):
        g = per_q[q]
        gid, reps = grouped[qi]
        gc = np.array(gcorrect[qi], dtype=bool)
        idx = g.index.to_numpy()
        lp = [to_arr(x) for x in g[col["logp"]]]
        kl = [to_arr(x) for x in g[col["kl"]]]
        rec = {"rollout_ids": g[c_rid].to_numpy(), "group": np.array(gid), "reps": reps, "group_correct": gc,
               "correct": gc[np.array(gid)], "ntok": cost[idx].astype(np.int64),
               "T": np.array([len(x) for x in lp]),
               "L": np.array([x.mean() if len(x) else -np.inf for x in lp]),
               "K": np.array([x.mean() if len(x) else -np.inf for x in kl]),
               "kl": [x.astype(np.float32) for x in kl], "gold": str(golds[qi]),
               "no_answer_group": reps.index(A.NO_ANSWER) if A.NO_ANSWER in reps else -1}
        if col["text"]:
            rec["text"] = g[col["text"]].tolist()
        if col["question"]:
            rec["question"] = g[col["question"]].iloc[0]
        if dcfg["kind"] == "vlm":
            vt = []
            for a_i, a_p, a_g in zip(g[col["attn_image"]], g[col["attn_prompt"]], g[col["attn_generated"]]):
                a_i, a_p, a_g = to_arr(a_i), to_arr(a_p), to_arr(a_g)
                den = a_i + a_p + a_g
                vt.append(np.where(den > 0, a_i / np.where(den > 0, den, 1), np.nan).astype(np.float32))
            rec["v"] = vt
            rec["V"] = np.array([np.nanmean(x) if len(x) and np.isfinite(x).any() else 0.0 for x in vt])
        pool[q] = rec

    n_roll = {len(p["group"]) for p in pool.values()}
    assert len(n_roll) == 1, f"unequal rollouts per question: {n_roll}"
    n_roll = n_roll.pop()
    splits = make_splits(qids, cfg)
    orders = make_orders(qids, n_roll, cfg)
    groups = np.array([len(p["reps"]) for p in pool.values()])
    checks.update({"rollouts_per_question": n_roll, "mean_distinct_groups_32": float(groups.mean()),
                   "no_answer_rate": float(np.mean([(p["group"] == p["no_answer_group"]).mean() for p in pool.values()])),
                   "rollout_accuracy": float(np.mean([p["correct"].mean() for p in pool.values()])),
                   "oracle_32": float(np.mean([p["correct"].any() for p in pool.values()]))})

    od = out_root() / "task3" / ds
    od.mkdir(parents=True, exist_ok=True)
    with open(od / "pool.pkl", "wb") as f:
        pickle.dump({"qids": qids, "pool": pool}, f)
    rd = results_root() / "task3" / ds
    save_json(splits, rd / "splits.json")
    np.savez_compressed(rd / "orders.npz", qids=np.array(qids), orders=orders)
    save_json(checks, rd / "data_checks.json")
    # per-question group table for auditing grouping/grading
    save_json({q: {"reps": pool[q]["reps"], "group_correct": pool[q]["group_correct"].tolist(), "gold": pool[q]["gold"],
                   "counts": np.bincount(pool[q]["group"], minlength=len(pool[q]["reps"])).tolist()} for q in qids},
              rd / "answer_groups.json")
    print({k: v for k, v in checks.items() if k != "columns_used"})


def load_pool(ds):
    with open(out_root() / "task3" / ds / "pool.pkl", "rb") as f:
        d = pickle.load(f)
    z = np.load(results_root() / "task3" / ds / "orders.npz", allow_pickle=True)
    assert list(z["qids"]) == d["qids"]
    return d["qids"], d["pool"], z["orders"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["math", "mathvista"])
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    cfg = load_config(CFG_PATH)
    if args.inspect:
        inspect(cfg, args.dataset)
    else:
        build(cfg, args.dataset, args.workers)


if __name__ == "__main__":
    main()
