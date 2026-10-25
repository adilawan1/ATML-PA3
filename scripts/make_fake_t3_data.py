"""Synthetic rollout files with the expected schema, for smoke-testing Task 3 code (NOT real data)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1])
rng = np.random.default_rng(0)
nq, nr = int(sys.argv[2]) if len(sys.argv) > 2 else 40, 32
for ds, folder, vlm in [("math", "MATH", False), ("mathvista", "MathVista", True)]:
    gen, prob, att, meta = [], [], [], {}
    for q in range(nq):
        sid = f"{q}" if not vlm else str(q + 1)
        if vlm:
            mc = q % 2 == 0
            choices = ["red", "blue", "green", "yellow"] if mc else None
            gold = "blue" if mc else str(q % 7)
            meta[sid] = {"pid": sid, "question": "?", "choices": choices, "answer": gold,
                         "question_type": "multi_choice" if mc else "free_form",
                         "answer_type": "text" if mc else "integer", "precision": None, "unit": None}
            pool = (["B", "A", "(C) green", "blue", None] if mc else [gold, str(q % 7 + 1), "x", None])
        else:
            gold = r"\frac{%d}{2}" % (q % 5 + 1)
            pool = [gold, f"{(q % 5 + 1) / 2}", f"{q % 5 + 2}", None, r"\sqrt{2}"]
        p_right = rng.uniform(0.1, 0.9)
        for r in range(nr):
            right = rng.random() < p_right
            ans = pool[0] if right else pool[rng.integers(1, len(pool))]
            T = int(rng.integers(20, 300))
            lp = (-np.abs(rng.normal(0.3 if right else 0.5, 0.3, T))).tolist()
            kl = np.abs(rng.normal(8 if right else 7, 2, T)).tolist()
            gen.append({"dataset": ds, "sample_id": sid, "rollout_id": r, "response": f"... \boxed{{{ans}}}",
                        "token_ids": list(range(T + 1)), "num_tokens": T, "extracted_answer": ans,
                        "gold_answer": gold, "problem": f"Problem {sid}"})
            prob.append({"dataset": ds, "sample_id": sid, "rollout_id": r, "logp": lp, "kl_u_p": kl})
            if vlm:
                att.append({"dataset": ds, "sample_id": sid, "rollout_id": r,
                            "attn_image": rng.uniform(0.05, 0.3, T).tolist(), "attn_prompt": rng.uniform(0.1, 0.3, T).tolist(),
                            "attn_generated": rng.uniform(0.3, 0.6, T).tolist()})
    d = root / folder
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(gen).to_parquet(d / "generations.parquet")
    pd.DataFrame(prob).to_parquet(d / "probability_metrics.parquet")
    if vlm:
        pd.DataFrame(att).to_parquet(d / "attention_metrics.parquet")
        json.dump(meta, open(root / "mathvista_meta.json", "w"))
print("fake data written to", root)
