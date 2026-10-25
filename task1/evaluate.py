"""Benchmark evaluation with lm-evaluation-harness (fixed task definitions, complete splits, zero-shot, FP16).

GSM8K: gsm8k_cot, num_fewshot=0, non-thinking chat template, greedy, max_gen_toks=1024, stock stops/filters.
HellaSwag / BoolQ / ARC-Easy: stock completion prompts, no chat template, likelihood scoring.

lm-eval's chat-template call does not forward `enable_thinking`, so small subclasses render the template
with enable_thinking=False; the rendered GSM8K prompt is checked for the empty think block.

  python -m task1.evaluate --condition lora16 --model_dir /content/merged/lora16
"""
import argparse
import gzip
import json
import time
from importlib.metadata import version

from transformers import AutoTokenizer

from common.utils import REPO_ROOT, gpu_info, load_config, results_root, save_json


def _no_think(self, chat_history, add_generation_prompt=True, **kw):
    return self.tokenizer.apply_chat_template(chat_history, tokenize=False,
                                              add_generation_prompt=add_generation_prompt,
                                              continue_final_message=not add_generation_prompt,
                                              enable_thinking=False)


def make_lm(backend, model_dir, batch_size, seed):
    if backend == "vllm":
        from lm_eval.models.vllm_causallms import VLLM

        cls = type("VLLMNoThink", (VLLM,), {"apply_chat_template": _no_think})
        return cls(pretrained=model_dir, dtype="float16", seed=seed, gpu_memory_utilization=0.85,
                   max_model_len=4096, batch_size=batch_size)
    from lm_eval.models.huggingface import HFLM

    cls = type("HFLMNoThink", (HFLM,), {"apply_chat_template": _no_think})
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return cls(pretrained=model_dir, dtype="float16" if device == "cuda" else "float32", batch_size=batch_size,
               device=device)


def run(lm, tasks, chat, gen_kwargs, seed):
    import lm_eval

    return lm_eval.simple_evaluate(model=lm, tasks=tasks, num_fewshot=0, apply_chat_template=chat,
                                   gen_kwargs=gen_kwargs, log_samples=True, random_seed=seed,
                                   numpy_random_seed=seed, torch_random_seed=seed, fewshot_random_seed=seed)


def gsm8k_rows(samples, tok, cap):
    rows = []
    for s in samples:
        resp = s["resps"][0][0]
        n = len(tok(resp, add_special_tokens=False)["input_ids"])
        # lm-eval logs one sample entry per (doc, filter); keep the filter name with each row
        rows.append({"doc_id": s["doc_id"], "filter": s.get("filter"), "target": s["target"],
                     "response": resp, "extracted": s["filtered_resps"][0] if s.get("filtered_resps") else None,
                     "exact_match": s.get("exact_match"), "gen_tokens": n, "truncated": n >= cap - 1,
                     "prompt": s["arguments"][0][0] if "arguments" in s else None})
    return rows


def likelihood_rows(samples):
    rows = []
    for s in samples:
        rows.append({"doc_id": s["doc_id"], "target": s["target"], "acc": s.get("acc"), "acc_norm": s.get("acc_norm"),
                     "loglik": [r[0][0] for r in s["resps"]], "context": s["arguments"][0][0]})
    return rows


def write_gz(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs/task1.yaml"))
    ap.add_argument("--only", choices=["gsm8k", "likelihood"], default=None)
    ap.add_argument("--limit", type=int, default=None, help="debug only; never for reported numbers")
    args = ap.parse_args()
    cfg, ec = load_config(args.config), load_config(args.config)["eval"]
    seed = cfg["seed"]
    out = results_root() / "task1/eval" / args.condition
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    meta = {"condition": args.condition, "model_dir": args.model_dir, "lm_eval_version": version("lm_eval"),
            **gpu_info(), "limit": args.limit}
    summary_path = out / "scores.json"
    scores = json.load(open(summary_path)) if summary_path.exists() else {}

    if args.only in (None, "gsm8k"):
        lm = make_lm(ec["backend"], args.model_dir, ec.get("gsm8k_batch_size", ec["batch_size"]), seed)
        t0 = time.perf_counter()
        r = run_limited(lm, [ec["gsm8k_task"]], True, f"max_gen_toks={ec['gsm8k_max_gen_toks']}", seed, args.limit)
        dt = time.perf_counter() - t0
        task = ec["gsm8k_task"]
        samples = r["samples"][task]
        rows = gsm8k_rows(samples, tok, ec["gsm8k_max_gen_toks"])
        assert "</think>" in rows[0]["prompt"], "non-thinking chat template was not applied"
        write_gz(rows, out / "gsm8k_samples.jsonl.gz")
        res = r["results"][task]
        flex = [x for x in rows if x["filter"] == "flexible-extract"]
        scores["gsm8k"] = {
            "flexible_em": res.get("exact_match,flexible-extract"), "strict_em": res.get("exact_match,strict-match"),
            "flexible_em_stderr": res.get("exact_match_stderr,flexible-extract"),
            "n": len(flex), "truncation_rate": sum(x["truncated"] for x in flex) / max(1, len(flex)),
            "mean_gen_tokens": sum(x["gen_tokens"] for x in flex) / max(1, len(flex)),
            "unextracted_rate": sum(x["extracted"] in (None, "", "[invalid]") for x in flex) / max(1, len(flex)),
            "time_s": dt, "backend": ec["backend"], "task_config": r["configs"][task].get("generation_kwargs"),
        }
        del lm
        _free()

    if args.only in (None, "likelihood"):
        lm = make_lm("hf", args.model_dir, ec["batch_size"], seed)
        t0 = time.perf_counter()
        r = run_limited(lm, ec["likelihood_tasks"], False, None, seed, args.limit)
        dt = time.perf_counter() - t0
        for task in ec["likelihood_tasks"]:
            res = r["results"][task]
            scores[task] = {"acc": res.get("acc,none"), "acc_norm": res.get("acc_norm,none"),
                            "acc_stderr": res.get("acc_stderr,none"), "acc_norm_stderr": res.get("acc_norm_stderr,none"),
                            "n": len(r["samples"][task])}
            write_gz(likelihood_rows(r["samples"][task]), out / f"{task}_samples.jsonl.gz")
        scores["likelihood_time_s"] = dt
        del lm
        _free()

    scores["meta"] = meta
    save_json(scores, summary_path)
    print(json.dumps({k: v for k, v in scores.items() if k != "meta"}, indent=1, default=str))


def run_limited(lm, tasks, chat, gen_kwargs, seed, limit):
    if limit is None:
        return run(lm, tasks, chat, gen_kwargs, seed)
    import lm_eval

    return lm_eval.simple_evaluate(model=lm, tasks=tasks, num_fewshot=0, apply_chat_template=chat,
                                   gen_kwargs=gen_kwargs, log_samples=True, limit=limit, random_seed=seed,
                                   numpy_random_seed=seed, torch_random_seed=seed, fewshot_random_seed=seed)


def _free():
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
