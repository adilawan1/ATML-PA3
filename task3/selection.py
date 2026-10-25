"""Candidate scoring and selection rules on a revealed candidate prefix (no gold answers, no unseen rollouts).

Ties: score/vote ties go to the earliest supporting rollout in the current order; RWC uses average ranks.
At N = 1 every rule returns the only candidate.
"""
import numpy as np
from scipy.stats import rankdata

LLM_METHODS = ["first", "sc", "max_L", "max_K", "rwc_K", "rwc_L", "shortest"]
VLM_METHODS = LLM_METHODS + ["max_V", "rwc_V", "max_VisConf", "rwc_VisConf", "max_Ztok", "rwc_Ztok"]
REQUIRED = {"llm": ["first", "sc", "max_L", "max_K", "rwc_K"],
            "vlm": ["first", "sc", "max_L", "max_K", "rwc_K", "max_V", "rwc_V", "max_VisConf", "rwc_VisConf"]}
LABELS = {"first": "FirstAns", "sc": "Self-consistency", "max_L": "Max log-prob", "max_K": "Max self-certainty",
          "rwc_K": "Self-certainty + RWC", "rwc_L": "Log-prob + RWC", "shortest": "Shortest rollout",
          "max_V": "Max vision", "rwc_V": "Vision + RWC", "max_VisConf": "Max VisConf", "rwc_VisConf": "VisConf + RWC",
          "max_Ztok": "Max tokenwise fusion", "rwc_Ztok": "Tokenwise fusion + RWC"}


def _finite(s):
    s = np.asarray(s, dtype=np.float64)
    return np.where(np.isfinite(s), s, -np.inf)


def argmax_earliest(scores):
    """Index (into the ordered prefix) of the highest score; np.argmax returns the first, i.e. earliest, maximum."""
    return int(np.argmax(_finite(scores)))


def vote(groups, weights):
    """Group with the largest total weight; ties broken by the earliest supporting rollout in the order."""
    groups = np.asarray(groups)
    uniq, first = np.unique(groups, return_index=True)
    W = np.array([weights[groups == g].sum() for g in uniq])
    best = W.max()
    tied = np.isclose(W, best, rtol=1e-9, atol=1e-12)
    cand = uniq[tied]
    return int(cand[np.argmin(first[tied])]), dict(zip(uniq.tolist(), W.tolist()))


def rwc_weights(scores):
    N = len(scores)
    if N == 1:
        return np.ones(1)
    r = rankdata(-_finite(scores), method="average")  # rank 1 = best; ties get average ranks
    return (N - r) / (N - 1)


def visconf_scores(p, cand, q=(0.05, 0.95)):
    """VisConf calibration recomputed on the current candidate set only. Returns (C, Z, Ztok) per candidate."""
    toks = [p["kl"][i] for i in cand]
    pooled = np.concatenate(toks) if any(len(t) for t in toks) else np.zeros(1)
    q05, q95 = np.percentile(pooled, [100 * q[0], 100 * q[1]])  # linear interpolation (NumPy default)
    den = max(q95 - q05, 1e-8)
    C, Ztok = np.full(len(cand), -np.inf), np.full(len(cand), -np.inf)
    for j, (i, k) in enumerate(zip(cand, toks)):
        if len(k) == 0:
            continue
        c = np.clip((k - q05) / den, 0.0, 1.0)
        C[j] = c.mean()
        v = p["v"][i]
        cv = c * v
        Ztok[j] = np.nanmean(cv) if np.isfinite(cv).any() else 0.0
    V = p["V"][cand]
    return C, V * C, Ztok


def candidate_scores(p, cand, kind):
    s = {"L": p["L"][cand], "K": p["K"][cand]}
    if kind == "vlm":
        s["V"] = p["V"][cand]
        _, s["VisConf"], s["Ztok"] = visconf_scores(p, cand)
    return s


def select(p, cand, method, scores=None, kind="llm"):
    """Return the selected answer-group id for one method on the ordered candidate prefix `cand`."""
    cand = np.asarray(cand)
    g = p["group"][cand]
    if len(cand) == 1 or method == "first":
        return int(g[0])
    if method == "sc":
        return vote(g, np.ones(len(cand)))[0]
    if method == "shortest":
        return int(g[int(np.argmin(p["ntok"][cand]))])
    if scores is None:
        scores = candidate_scores(p, cand, kind)
    agg, name = method.split("_", 1)
    s = scores[name]
    if agg == "max":
        return int(g[argmax_earliest(s)])
    return vote(g, rwc_weights(s))[0]


def rwc_margin(p, cand, score):
    """(top - runner-up) / total RWC vote weight; runner-up = 0 when only one answer group exists."""
    g = p["group"][np.asarray(cand)]
    w = rwc_weights(score)
    winner, W = vote(g, w)
    vals = sorted(W.values(), reverse=True)
    total = sum(vals)
    second = vals[1] if len(vals) > 1 else 0.0
    return winner, (vals[0] - second) / total if total > 0 else 0.0
