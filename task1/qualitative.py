"""Greedy qualitative generations: 24 prompts x 7 conditions = 168 responses (thinking off, 512-token cap).

  python -m task1.qualitative --condition lora16 --model_dir /content/merged/lora16
  python -m task1.qualitative --collect     # merge per-condition files, build the 42-response annotation sheet

Automatic flags support (not replace) the manual annotation: JSON validity, template-field compliance,
truncation, repetition (repeated 4-gram fraction), and math-framing keywords.
"""
import argparse
import csv
import json
import re
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.utils import REPO_ROOT, load_config, load_json, results_root, save_json, set_seed

MATH_WORDS = re.compile(r"\\boxed|\$\$|\\frac|equation|calculate|solve for|step \d|= \d|\bx\s*=|percent|total of|\d+\s*[\+\-\*/×]\s*\d+")


def load_prompts(path):
    entries = yaml.safe_load(open(path, encoding="utf-8"))
    out = []
    for e in entries:
        qid = e["id"]
        fmt = "json" if qid.endswith("_json") else "template" if qid.endswith("_template") else "plain"
        base = qid.removesuffix("_json").removesuffix("_template")
        out.append({"id": qid, "question": base, "format": fmt, "system": e.get("system"),
                    "prompt": e["paraphrases"][0]})
    return out


def flags(rec, tok, cap):
    text = rec["response"]
    n = len(tok(text, add_special_tokens=False)["input_ids"])
    f = {"gen_tokens": n, "truncated": rec.get("finished") is False}
    if rec["format"] == "json":
        s = text.strip()
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
        try:
            json.loads(s)
            f["json_valid"] = True
        except Exception:
            f["json_valid"] = False
    if rec["format"] == "template":
        fields = re.findall(r"^(\w+)\s*=", rec["prompt"], flags=re.M)
        f["template_fields_present"] = sum(bool(re.search(rf"^\s*{fld}\s*=", text, flags=re.M)) for fld in fields)
        f["template_fields_total"] = len(fields)
    words = text.split()
    grams = [" ".join(words[i:i + 4]) for i in range(max(0, len(words) - 3))]
    f["repeat_4gram_frac"] = 1 - len(set(grams)) / len(grams) if grams else 0.0
    f["math_markers"] = len(MATH_WORDS.findall(text))
    return f


@torch.no_grad()
def generate(cfg, condition, model_dir):
    qc = cfg["qualitative"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
    prompts = load_prompts(REPO_ROOT / qc["file"])
    set_seed(cfg["seed"])
    recs = []
    for p in prompts:
        msgs = ([{"role": "system", "content": p["system"]}] if p["system"] else []) + [{"role": "user", "content": p["prompt"]}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
        out = model.generate(**enc, max_new_tokens=qc["max_new_tokens"], do_sample=False, temperature=None,
                             top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
        gen = out[0, enc["input_ids"].shape[1]:]
        finished = bool((gen == tok.convert_tokens_to_ids("<|im_end|>")).any() or (gen == tok.eos_token_id).any())
        rec = {"condition": condition, **p, "response": tok.decode(gen, skip_special_tokens=True), "finished": finished}
        rec.update(flags(rec, tok, qc["max_new_tokens"]))
        recs.append(rec)
        print(f"[{condition}] {p['id']}: {rec['response'][:80]!r}")
    save_json(recs, results_root() / "task1/qualitative" / f"{condition}.json")


def collect(cfg):
    qc = cfg["qualitative"]
    order = [c["name"] for c in cfg["eval"]["conditions"]]
    d = results_root() / "task1/qualitative"
    allr = []
    for c in order:
        p = d / f"{c}.json"
        if p.exists():
            allr += load_json(p)
    grouped = {}
    for r in allr:
        grouped.setdefault(r["question"], {}).setdefault(r["format"], {})[r["condition"]] = r["response"]
    save_json(grouped, d / "all_responses_grouped.json")
    # 42-response annotation sheet: first & last plain questions x 3 formats x 7 conditions
    sel = [r for r in allr if r["question"] in qc["annotate_ids"]]
    sel.sort(key=lambda r: (qc["annotate_ids"].index(r["question"]), ["plain", "json", "template"].index(r["format"]),
                            order.index(r["condition"])))
    cols = ["question", "format", "condition", "truncated", "gen_tokens", "json_valid", "template_fields_present",
            "template_fields_total", "repeat_4gram_frac", "math_markers",
            "ann_format_compliance", "ann_coherence_relevance", "ann_repetition", "ann_math_framing",
            "ann_inappropriate", "ann_excerpt", "ann_notes", "response"]
    with open(d / "annotation_sheet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sel:
            w.writerow(r)
    # automatic flag summary per condition over all 24 prompts
    summ = {}
    for c in order:
        rs = [r for r in allr if r["condition"] == c]
        if not rs:
            continue
        js = [r for r in rs if r["format"] == "json"]
        tp = [r for r in rs if r["format"] == "template"]
        summ[c] = {"n": len(rs), "truncated": sum(r["truncated"] for r in rs),
                   "json_valid": sum(r.get("json_valid", False) for r in js), "json_n": len(js),
                   "template_full": sum(r["template_fields_present"] == r["template_fields_total"] for r in tp),
                   "template_n": len(tp), "mean_repeat_4gram": sum(r["repeat_4gram_frac"] for r in rs) / len(rs),
                   "responses_with_math_markers": sum(r["math_markers"] > 0 for r in rs),
                   "mean_gen_tokens": sum(r["gen_tokens"] for r in rs) / len(rs)}
    save_json(summ, d / "auto_flags_summary.json")
    print(json.dumps(summ, indent=1))
    print(f"annotation sheet: {len(sel)} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs/task1.yaml"))
    ap.add_argument("--condition")
    ap.add_argument("--model_dir")
    ap.add_argument("--collect", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.collect:
        collect(cfg)
    else:
        generate(cfg, args.condition, args.model_dir)


if __name__ == "__main__":
    main()
