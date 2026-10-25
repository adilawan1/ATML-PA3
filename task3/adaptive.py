"""Step 5: adaptive sampling (offline simulation of stopping between completed rollouts).

Rollouts are revealed in the saved orders; stopping is checked after n in {4, 8, 16}, otherwise n = 32.
Both controllers return the current self-consistency answer.
  agreement:      unique non-NO_ANSWER leading answer with vote share >= tau
  quality-aware:  agreement AND SC winner == RWC winner AND RWC margin >= gamma
                  (self-certainty for MATH, VisConf for MathVista; scores/calibration/ranks on revealed rollouts only)
Settings are chosen on dev (lowest mean cost within 1 pp of fixed-32 SC; ties: higher acc, smaller tau, gamma),
frozen, then evaluated on the held-out split.

  python -m task3.adaptive --dataset math
"""
import argparse
import itertools

import numpy as np
import pandas as pd

from common.utils import REPO_ROOT, load_config, load_json, results_root, save_json
from task3.data import load_pool
from task3.offline import paired_bootstrap
from task3.selection import candidate_scores, rwc_margin, vote

CFG = load_config(REPO_ROOT / "configs/task3.yaml")


def precompute(qids, pool, orders, kind):
    """Stopping statistics at every check point, for every (question, order)."""
    ac = CFG["adaptive"]
    qscore = "VisConf" if kind == "vlm" else "K"
    rows = []
    for qi, q in enumerate(qids):
        p = pool[q]
        for o in range(orders.shape[1]):
            perm = orders[qi, o]
            r = {"qid": q, "order": o, "frac_correct_pool": float(p["correct"].mean())}
            for n in ac["checks"] + [ac["max_n"]]:
                cand = perm[:n]
                g = p["group"][cand]
                counts = np.bincount(g, minlength=len(p["reps"]))
                top = counts.max()
                sc = vote(g, np.ones(n))[0]
                r[f"sc_{n}"] = sc
                r[f"sc_correct_{n}"] = bool(p["group_correct"][sc])
                r[f"cost_{n}"] = int(p["ntok"][cand].sum())
                if n == ac["max_n"]:
                    continue
                r[f"unique_{n}"] = bool((counts == top).sum() == 1)
                r[f"share_{n}"] = top / n
                r[f"noans_{n}"] = bool(sc == p["no_answer_group"])
                s = candidate_scores(p, cand, kind)[qscore]
                rw, margin = rwc_margin(p, cand, s)
                r[f"rwc_agree_{n}"] = bool(rw == sc)
                r[f"margin_{n}"] = margin
            rows.append(r)
    return pd.DataFrame(rows)


def simulate(df, family, tau, gamma):
    ac = CFG["adaptive"]
    stop = np.full(len(df), ac["max_n"])
    done = np.zeros(len(df), dtype=bool)
    if family != "fixed32":
        for n in ac["checks"]:
            ok = df[f"unique_{n}"] & ~df[f"noans_{n}"] & (df[f"share_{n}"] >= tau - 1e-12)
            if family == "quality":
                ok &= df[f"rwc_agree_{n}"] & (df[f"margin_{n}"] >= gamma - 1e-12)
            new = ok.to_numpy() & ~done
            stop[new] = n
            done |= new
    correct = np.array([df[f"sc_correct_{n}"].iat[i] for i, n in enumerate(stop)])
    cost = np.array([df[f"cost_{n}"].iat[i] for i, n in enumerate(stop)])
    out = df[["qid", "order", "frac_correct_pool", f"sc_correct_{ac['max_n']}"]].copy()
    out["stop_n"], out["correct"], out["cost"] = stop, correct, cost
    return out


def qmean(sim):
    """Average over orders within question, then across questions."""
    g = sim.groupby("qid").agg(acc=("correct", "mean"), cost=("cost", "mean"), n=("stop_n", "mean"))
    return g.acc.mean(), g.cost.mean(), g.n.mean(), g


def settings():
    ac = CFG["adaptive"]
    fams = {"agreement": [(t, None) for t in ac["taus"]],
            "quality": [(t, g) for t, g in itertools.product(ac["taus"], ac["gammas"])]}
    return fams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["math", "mathvista"])
    args = ap.parse_args()
    ds, kind = args.dataset, CFG["datasets"][args.dataset]["kind"]
    rd = results_root() / "task3" / ds
    qids, pool, orders = load_pool(ds)
    splits = load_json(rd / "splits.json")
    pre = precompute(qids, pool, orders, kind)
    dev, ev = pre[pre.qid.isin(splits["dev"])], pre[pre.qid.isin(splits["eval"])]
    acc32, cost32, _, _ = qmean(simulate(dev, "fixed32", None, None))
    tol = CFG["adaptive"]["tolerance_pp"] / 100

    grid, chosen = [], {}
    for fam, sets in settings().items():
        cands = [{"family": fam, "tau": None, "gamma": None, "fallback32": True, "acc": acc32, "cost": cost32}]
        for tau, gamma in sets:
            a, c, n, _ = qmean(simulate(dev, fam, tau, gamma))
            cands.append({"family": fam, "tau": tau, "gamma": gamma, "fallback32": False, "acc": a, "cost": c, "mean_n": n})
        grid += cands
        ok = [c for c in cands if c["acc"] >= acc32 - tol - 1e-12]
        ok.sort(key=lambda c: (c["cost"], -c["acc"], c["tau"] if c["tau"] is not None else 9,
                               c["gamma"] if c["gamma"] is not None else 9))
        chosen[fam] = ok[0]
    print("dev fixed-32 SC acc", round(100 * acc32, 2), "| chosen:", chosen)

    # held-out evaluation with frozen settings
    held = {}
    a32, c32, _, g32 = qmean(simulate(ev, "fixed32", None, None))
    held["fixed32_sc"] = {"acc": a32, "cost": c32, "mean_rollouts": 32.0}
    sims = {}
    for fam, ch in chosen.items():
        sim = simulate(ev, "fixed32" if ch["fallback32"] else fam, ch["tau"], ch["gamma"])
        sims[fam] = sim
        a, c, n, g = qmean(sim)
        dist = sim.stop_n.value_counts(normalize=True).sort_index()
        early = sim[sim.stop_n < CFG["adaptive"]["max_n"]]
        wrong_early = early[~early.correct]
        rec = wrong_early[wrong_early[f"sc_correct_{CFG['adaptive']['max_n']}"]]
        held[fam] = {"setting": ch, "acc": a, "cost": c, "mean_rollouts": n,
                     "stop_distribution": {str(int(k)): float(v) for k, v in dist.items()},
                     "early_stop_rate": float(len(early) / len(sim)),
                     "incorrect_early_stops": int(len(wrong_early)),
                     "incorrect_early_stops_rate": float(len(wrong_early) / len(sim)),
                     "incorrect_early_recoverable_by_fixed32": int(len(rec)),
                     "bootstrap_vs_fixed32": paired_bootstrap(g.acc.loc[g32.index], g32.acc, CFG["offline"]["bootstrap"],
                                                              CFG["seed"]),
                     "cost_saving_vs_fixed32": 1 - c / c32}
        # retrospective: stopping vs fraction of correct candidates in the full pool (gold used for analysis only)
        bins = pd.cut(sim.frac_correct_pool, [-0.01, 0, 0.25, 0.5, 0.75, 0.99, 1.0])
        held[fam]["by_pool_correct_fraction"] = (sim.groupby(bins, observed=True)
                                                 .agg(mean_stop=("stop_n", "mean"), acc=("correct", "mean"),
                                                      n=("stop_n", "size")).reset_index()
                                                 .assign(frac_correct_pool=lambda d: d.frac_correct_pool.astype(str))
                                                 .to_dict(orient="records"))
        sim.to_csv(rd / f"adaptive_{fam}_eval_rows.csv.gz", index=False)
    # incorrect early-stop examples, fixed rule: first 5 by (qid order in split, order index)
    ex = []
    for fam, sim in sims.items():
        s = sim[(sim.stop_n < 32) & ~sim.correct].copy()
        s["rank"] = s.qid.map({q: i for i, q in enumerate(splits["eval"])})
        for _, r in s.sort_values(["rank", "order"]).head(5).iterrows():
            p = pool[r.qid]
            row = pre[(pre.qid == r.qid) & (pre.order == r.order)].iloc[0]
            ex.append({"family": fam, "qid": r.qid, "order": int(r.order), "stop_n": int(r.stop_n),
                       "answer_at_stop": p["reps"][int(row[f"sc_{int(r.stop_n)}"])],
                       "answer_at_32": p["reps"][int(row["sc_32"])], "gold": p["gold"],
                       "share_at_stop": float(row[f"share_{int(r.stop_n)}"]),
                       "frac_correct_pool": float(r.frac_correct_pool)})
    save_json({"dev_fixed32_acc": acc32, "dev_fixed32_cost": cost32, "grid_dev": grid, "chosen": chosen,
               "heldout": held, "incorrect_early_stop_examples": ex}, rd / "adaptive.json")
    for k, v in held.items():
        print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()
                  if kk in ("acc", "cost", "mean_rollouts", "stop_distribution", "incorrect_early_stops")})


if __name__ == "__main__":
    main()
