"""Unit tests for Task 3 selection rules (run: python -m pytest tests -q, or python -m tests.test_task3_selection)."""
import numpy as np

from task3.answers import extract_last_boxed
from task3.selection import rwc_margin, rwc_weights, select, vote


def pool(groups, L=None, K=None, ntok=None):
    n = len(groups)
    return {"group": np.array(groups), "L": np.array(L if L is not None else [0.0] * n),
            "K": np.array(K if K is not None else [0.0] * n), "ntok": np.array(ntok if ntok is not None else [10] * n)}


def test_vote_tie_goes_to_earliest():
    # groups 1 and 0 tie with 2 votes each; group 1 appears first in the order
    assert vote(np.array([1, 0, 0, 1]), np.ones(4))[0] == 1


def test_sc_respects_order_not_ids():
    p = pool([0, 1, 1, 0])
    assert select(p, np.array([1, 0, 2, 3]), "sc") == 1   # order starts with rollout 1 (group 1)
    assert select(p, np.array([0, 1, 2, 3]), "sc") == 0


def test_max_score_tie_earliest():
    p = pool([0, 1, 2], K=[1.0, 2.0, 2.0])
    assert select(p, np.array([2, 1, 0]), "max_K") == 2
    assert select(p, np.array([1, 2, 0]), "max_K") == 1


def test_rwc_average_ranks():
    w = rwc_weights(np.array([3.0, 1.0, 1.0, 0.0]))
    # ranks: 1, 2.5, 2.5, 4 -> w = (4-r)/3
    assert np.allclose(w, [1.0, 0.5, 0.5, 0.0])


def test_rwc_can_beat_majority():
    p = pool([0, 1, 1, 2, 2], K=[10, 1, 2, 3, 0])
    # weights by rank: K=10 ->1, 3 ->0.75, 2 ->0.5, 1 ->0.25, 0 ->0
    # group0: 1.0, group1: 0.25+0.5=0.75, group2: 0.75+0 = 0.75
    assert select(p, np.arange(5), "rwc_K") == 0
    assert select(p, np.arange(5), "sc") == 1


def test_margin_single_group():
    p = pool([3, 3, 3, 3])
    win, m = rwc_margin(p, np.arange(4), np.array([1.0, 2, 3, 4]))
    assert win == 3 and m == 1.0


def test_n1_returns_only_candidate():
    p = pool([5, 6], K=[0, 9])
    for m in ("sc", "max_K", "rwc_K", "shortest", "first"):
        assert select(p, np.array([0]), m) == 5


def test_boxed_extraction():
    assert extract_last_boxed(r"a \boxed{1} then \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert extract_last_boxed(r"\boxed{\text{(A) }x^{2}}") == r"\text{(A) }x^{2}"
    assert extract_last_boxed("no box") is None
    assert extract_last_boxed(r"\boxed{unbalanced") is None


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
    print("all passed")
