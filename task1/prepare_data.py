"""Select and tokenize the first 40,000 eligible MetaMathQA examples (source order, no shuffling).

Eligibility: non-empty query and response; formatted length <= 2048 tokens; neither `query` nor
`original_question` matches a GSM8K test question after text normalization.

Format: prompt = chat template of the user turn with the non-thinking generation prompt
(`<|im_start|>assistant\n<think>\n\n</think>\n\n`, identical to the evaluation prefix);
target = response tokens + <|im_end|>. Only target tokens receive loss.

Outputs:
  $PA3_OUT/task1/data/train_tokens.npz   (flat int32 ids, offsets, prompt lengths)
  results/task1/data_selection.json      (source indices, exclusion counts, hash)
"""
import argparse
import hashlib

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from common.utils import REPO_ROOT, load_config, normalize_text, out_root, results_root, save_json


def build_prompt(tok, question):
    return tok.apply_chat_template([{"role": "user", "content": question}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs/task1.yaml"))
    ap.add_argument("--chunk", type=int, default=2000)
    args = ap.parse_args()
    cfg = load_config(args.config)
    n_target, max_len = cfg["data"]["n_examples"], cfg["data"]["max_len"]

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    eot = tok.convert_tokens_to_ids("<|im_end|>")
    gsm = load_dataset(cfg["data"]["gsm8k"], "main", split="test")
    gsm_norm = {normalize_text(q) for q in gsm["question"]}
    mm = load_dataset(cfg["data"]["source"], split="train")

    ids_flat, offsets, prompt_lens, src_idx = [], [0], [], []
    excl = {"empty": 0, "too_long": 0, "gsm8k_overlap": 0}
    scanned = 0
    for start in range(0, len(mm), args.chunk):
        rows = mm[start:start + args.chunk]
        cand = []
        for j, (q, r, oq) in enumerate(zip(rows["query"], rows["response"], rows["original_question"])):
            if not q or not q.strip() or not r or not r.strip():
                excl["empty"] += 1
                continue
            if normalize_text(q) in gsm_norm or (oq and normalize_text(oq) in gsm_norm):
                excl["gsm8k_overlap"] += 1
                continue
            cand.append((start + j, q, r))
        p_tok = tok([build_prompt(tok, q) for _, q, _ in cand], add_special_tokens=False)["input_ids"]
        r_tok = tok([r for _, _, r in cand], add_special_tokens=False)["input_ids"]
        done = False
        for (idx, _, _), p, r in zip(cand, p_tok, r_tok):
            seq = p + r + [eot]
            if len(seq) > max_len:
                excl["too_long"] += 1
                continue
            ids_flat.extend(seq)
            offsets.append(len(ids_flat))
            prompt_lens.append(len(p))
            src_idx.append(idx)
            if len(src_idx) == n_target:
                done = True
                scanned = idx + 1
                break
        if done:
            break
        print(f"scanned {start + args.chunk:,} rows, kept {len(src_idx):,}", flush=True)
    assert len(src_idx) == n_target, f"only {len(src_idx)} eligible examples"
    # exclusion counts above only cover rows before the last kept example
    ids_flat = np.asarray(ids_flat, dtype=np.int32)
    offsets = np.asarray(offsets, dtype=np.int64)
    prompt_lens = np.asarray(prompt_lens, dtype=np.int32)
    lengths = np.diff(offsets)
    out = out_root() / "task1/data"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "train_tokens.npz", ids=ids_flat, offsets=offsets, prompt_lens=prompt_lens,
             src_idx=np.asarray(src_idx, dtype=np.int64))
    digest = hashlib.sha256(ids_flat.tobytes() + offsets.tobytes() + prompt_lens.tobytes()).hexdigest()
    save_json({
        "model": cfg["model"], "n_examples": n_target, "rows_scanned": scanned, "excluded": excl,
        "token_hash_sha256": digest,
        "total_tokens": int(lengths.sum()), "response_tokens": int((lengths - prompt_lens).sum()),
        "len_mean": float(lengths.mean()), "len_p50": float(np.median(lengths)), "len_max": int(lengths.max()),
        "example_prompt": build_prompt(tok, "<QUESTION>"),
        "source_indices": src_idx,
    }, results_root() / "task1/data_selection.json")
    print("excluded:", excl, "| total tokens:", int(lengths.sum()), "| hash:", digest[:12])


if __name__ == "__main__":
    main()
