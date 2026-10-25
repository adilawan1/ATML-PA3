"""Fix the activation-probe set (32 GSM8K test + 32 HellaSwag validation) with seed 6304, before any results.

Prompts are rendered in the benchmark formats used by lm-eval:
  GSM8K (gsm8k_cot, zero-shot): user turn "Q: {question}\\nA:" through the non-thinking chat template.
  HellaSwag: preprocess(activity_label + ": " + ctx_a + " " + ctx_b.capitalize()), no chat template.
The diagnostics script cross-checks these strings against prompts logged by lm-eval for the baseline.
"""
import re

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from common.utils import REPO_ROOT, load_config, results_root, save_json


def hellaswag_preprocess(text):  # copied from lm-eval tasks/hellaswag/utils.py
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def gsm8k_prompt(tok, q):
    return tok.apply_chat_template([{"role": "user", "content": f"Q: {q}\nA:"}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def main():
    cfg = load_config(REPO_ROOT / "configs/task1.yaml")
    dc = cfg["diagnostics"]
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    rng = np.random.default_rng(cfg["seed"])
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    hs = load_dataset("Rowan/hellaswag", split="validation")
    gi = sorted(rng.choice(len(gsm), dc["n_probe_math"], replace=False).tolist())
    hi = sorted(rng.choice(len(hs), dc["n_probe_nonmath"], replace=False).tolist())
    probes = []
    for i in gi:
        probes.append({"set": "math", "dataset": "gsm8k_test", "doc_id": i, "prompt": gsm8k_prompt(tok, gsm[i]["question"])})
    for i in hi:
        d = hs[i]
        ctx = d["ctx_a"] + " " + d["ctx_b"].capitalize()
        probes.append({"set": "nonmath", "dataset": "hellaswag_validation", "doc_id": i,
                       "prompt": hellaswag_preprocess(d["activity_label"] + ": " + ctx)})
    save_json({"seed": cfg["seed"], "gsm8k_ids": gi, "hellaswag_ids": hi, "probes": probes},
              results_root() / "task1/probe_set.json")
    print(f"saved {len(probes)} probes")


if __name__ == "__main__":
    main()
