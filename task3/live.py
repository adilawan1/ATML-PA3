"""Step 4: parallel sampling vs one longer response vs sequential refinement on 50 MATH evaluation questions.

Per question (3,072-token output allowance each):
  single:   one response, max 3,072 new tokens
  parallel: three independent responses, max 1,024 each; self-consistency over their answers
  refine:   parallel response #1 (shared) + one critique-and-revision response (max 2,048); revised answer returned
FP16, no quantization, batch size 1, independently seeded calls, no gold feedback. Every call is saved
(prompt, response, seed, token counts, truncation, time, peak memory), and the run resumes by skipping saved calls.

  python -m task3.live            # generate
  python -m task3.live --summarize
"""
import argparse
import time

import numpy as np
import torch

from common.utils import (REPO_ROOT, append_jsonl, gpu_info, load_config, load_json, out_root, read_jsonl,
                          results_root, save_json, sync)
from task3 import answers as A
from task3.selection import vote

CFG = load_config(REPO_ROOT / "configs/task3.yaml")
LC = CFG["live"]
OUT = results_root() / "task3/live"


def problems():
    """(qid, problem text, gold) for the live split: from the committed live_problems.json, else the built pool."""
    lp = results_root() / "task3/math/live_problems.json"
    if lp.exists():
        d = load_json(lp)
        assert [p["qid"] for p in d["problems"]] == load_json(results_root() / "task3/math/splits.json")["live"]
        return [(p["qid"], p["question"], p["gold"]) for p in d["problems"]]
    import pickle

    splits = load_json(results_root() / "task3/math/splits.json")
    with open(out_root() / "task3/math/pool.pkl", "rb") as f:
        pool = pickle.load(f)["pool"]
    qs = splits["live"]
    missing = [q for q in qs if "question" not in pool[q]]
    if missing:
        raise SystemExit("problem text not found in the supplied generations; add its column to configs/task3.yaml")
    return [(q, pool[q]["question"], pool[q]["gold"]) for q in qs]


def user_prompt(problem):
    return problem + LC["separator"] + LC["instruction"]


def call(model, tok, messages, max_new, seed, device):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    sync()
    t0 = time.perf_counter()
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=LC["temperature"],
                         top_p=LC["top_p"], top_k=LC["top_k"], repetition_penalty=LC["repetition_penalty"],
                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
    sync()
    dt = time.perf_counter() - t0
    gen = out[0, enc["input_ids"].shape[1]:]
    stop_ids = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
    finished = any(int(t) in stop_ids for t in gen)
    n_out = int(sum(int(t) not in stop_ids for t in gen))  # termination token not counted
    resp = tok.decode(gen, skip_special_tokens=True)
    return {"prompt_text": text, "response": resp, "seed": seed, "input_tokens": int(enc["input_ids"].shape[1]),
            "output_tokens": n_out, "max_new_tokens": max_new, "truncated": (not finished) and len(gen) >= max_new,
            "time_s": dt, "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0,
            "extracted": A.extract_last_boxed(resp)}


def generate(model_path=None, limit=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = model_path or LC["model"]
    kw = {"revision": LC["revision"]} if (LC["revision"] and not model_path) else {}
    tok = AutoTokenizer.from_pretrained(name, **kw)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16 if device == "cuda" else torch.float32,
                                                 **kw).to(device).eval()
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / "calls.jsonl"
    done = {(r["qid"], r["call"]) for r in read_jsonl(log)} if log.exists() else set()
    save_json({"model": name, "revision": LC["revision"], "dtype": "float16", "batch_size": 1, **gpu_info(),
               "settings": LC, "seed": CFG["seed"]}, OUT / "run_meta.json")
    probs = problems()[:limit] if limit else problems()
    for qi, (q, prob, _) in enumerate(probs):
        base = [{"role": "system", "content": LC["system"]}, {"role": "user", "content": user_prompt(prob)}]
        seed0 = CFG["seed"] * 1000 + qi * 10
        prev = {r["call"]: r for r in read_jsonl(log) if r["qid"] == q} if log.exists() else {}
        plan = [(f"parallel_{k}", base, LC["parallel_max"], seed0 + k) for k in range(1, LC["parallel_n"] + 1)]
        plan += [("refine", None, LC["revision_max"], seed0 + 5), ("single", base, LC["single_max"], seed0 + 6)]
        with torch.no_grad():
            for name_c, msgs, mx, seed in plan:
                if (q, name_c) in done:
                    continue
                if name_c == "refine":
                    first = prev["parallel_1"]
                    msgs = base + [{"role": "assistant", "content": first["response"]},
                                   {"role": "user", "content": LC["review"]}]
                r = {"qid": q, "call": name_c, **call(model, tok, msgs, mx, seed, device)}
                append_jsonl(r, log)
                prev[name_c] = r
                print(f"[{qi + 1}/{len(probs)}] {q} {name_c}: out={r['output_tokens']} trunc={r['truncated']} "
                      f"ans={r['extracted']!s:.30} t={r['time_s']:.1f}s", flush=True)


def summarize():
    calls = read_jsonl(OUT / "calls.jsonl")
    gold = {q: g for q, _, g in problems()}
    mv = A.MathVerifier()
    by_q = {}
    for r in calls:
        by_q.setdefault(r["qid"], {})[r["call"]] = r
    for r in calls:
        a = r["extracted"]
        r["correct"] = (not A.is_missing(a)) and mv.correct(gold[r["qid"]], a)
    rows, trans = [], {"W->C": 0, "C->W": 0, "C->C": 0, "W->W": 0}
    for q, c in by_q.items():
        if not all(k in c for k in ("parallel_1", "parallel_2", "parallel_3", "refine", "single")):
            continue
        par = [c[f"parallel_{k}"] for k in (1, 2, 3)]
        gid, reps = A.group_math([p["extracted"] for p in par], mv)
        win = vote(np.array(gid), np.ones(3))[0]
        par_correct = reps[win] != A.NO_ANSWER and mv.correct(gold[q], reps[win])
        init_c, rev_c = c["parallel_1"]["correct"], c["refine"]["correct"]
        trans[("C" if init_c else "W") + "->" + ("C" if rev_c else "W")] += 1
        strat = {
            "single": ([c["single"]], c["single"]["correct"], c["single"]["extracted"]),
            "parallel": (par, par_correct, reps[win]),
            "refine": ([c["parallel_1"], c["refine"]], rev_c, c["refine"]["extracted"]),
        }
        for s, (cl, corr, ans) in strat.items():
            rows.append({"qid": q, "strategy": s, "correct": bool(corr), "answer": ans,
                         "input_tokens": sum(x["input_tokens"] for x in cl), "output_tokens": sum(x["output_tokens"] for x in cl),
                         "time_s": sum(x["time_s"] for x in cl), "any_truncated": any(x["truncated"] for x in cl),
                         "final_extracted": not A.is_missing(ans) and ans != A.NO_ANSWER,
                         "calls_truncated": sum(x["truncated"] for x in cl), "n_calls": len(cl),
                         "peak_mem_gb": max(x["peak_mem_gb"] for x in cl)})
    import pandas as pd

    df = pd.DataFrame(rows)
    summ = df.groupby("strategy").agg(acc=("correct", "mean"), input_tokens=("input_tokens", "mean"),
                                      output_tokens=("output_tokens", "mean"), time_s=("time_s", "mean"),
                                      final_extraction_rate=("final_extracted", "mean"),
                                      any_call_truncated_rate=("any_truncated", "mean"),
                                      call_truncation_rate=("calls_truncated", "sum"), n_calls=("n_calls", "sum"),
                                      peak_mem_gb=("peak_mem_gb", "max")).reset_index()
    summ["call_truncation_rate"] = summ.call_truncation_rate / summ.n_calls
    per_call = pd.DataFrame(calls).groupby("call").agg(acc=("correct", "mean"), out=("output_tokens", "mean"),
                                                       trunc=("truncated", "mean"),
                                                       extracted=("extracted", lambda x: x.notna().mean())).reset_index()
    boot = {}
    from task3.offline import paired_bootstrap

    piv = df.pivot(index="qid", columns="strategy", values="correct").astype(float)
    for a, b in [("parallel", "single"), ("refine", "single"), ("refine", "parallel")]:
        boot[f"{a}-{b}"] = paired_bootstrap(piv[a], piv[b], CFG["offline"]["bootstrap"], CFG["seed"])
    df.to_csv(OUT / "per_question.csv", index=False)
    save_json({"n_questions": int(df.qid.nunique()), "strategies": summ.to_dict(orient="records"),
               "per_call": per_call.to_dict(orient="records"), "refine_transitions": trans, "bootstrap": boot},
              OUT / "summary.json")
    print(summ.round(3).to_string())
    print(per_call.round(3).to_string())
    print("transitions vs shared initial answer:", trans)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--model", default=None, help="override (smoke tests only)")
    ap.add_argument("--limit", type=int, default=None, help="smoke tests only")
    args = ap.parse_args()
    if args.summarize:
        summarize()
    else:
        generate(args.model, args.limit)


if __name__ == "__main__":
    main()
