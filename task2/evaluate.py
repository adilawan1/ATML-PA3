"""Final grounding evaluation on both test suites under four image conditions.

Conditions (question and original references fixed): correct image, complementary swap (partner image),
mismatched (image from another pair of the same suite; fixed seeded derangement, saved), no image.
Greedy decoding, thinking disabled, 16-token cap, batch size 8.

Metrics per suite x condition: accuracy (grounding: normalized EM split presence/counting; VQA: soft
consensus accuracy split yes/no, number, other), pair accuracy, identical-answer rate, empty-answer rate,
mean reference NLL, generation time per answer, partner-follow rate; correct-image advantages over each
control in accuracy and NLL; 95% CIs from 4,000 bootstrap resamples of complete pairs (same resampled
pair IDs across conditions).

  python -m task2.evaluate [--run vis196] [--weights stage3_weights.pt]
"""
import argparse
import time

import numpy as np
import torch

from common import lora as L
from common.utils import REPO_ROOT, gpu_info, load_config, out_root, peak_mem_gb, results_root, save_json
from task2.data import load_all, pair_examples
from task2.vlm import FeatureStore, add_lora, build_vlm
from task2.vqa_norm import normalize, soft_accuracy

CONDS = ["correct", "complementary", "mismatched", "none"]


def derangement(n, seed):
    rng = np.random.default_rng(seed)
    while True:
        p = rng.permutation(n)
        if n < 2 or not (p == np.arange(n)).any():
            return p


def condition_examples(exs, perm):
    """Return {condition: (examples with show_image set, with_image flags)}; exs alternate slot 0/1 per pair."""
    by_pair = {}
    for e in exs:
        by_pair.setdefault(e["pair"], {})[e["slot"]] = e
    out = {"correct": ([dict(e) for e in exs], [True] * len(exs))}
    out["complementary"] = ([dict(e, show_image=by_pair[e["pair"]][1 - e["slot"]]["image_id"]) for e in exs], [True] * len(exs))
    out["mismatched"] = ([dict(e, show_image=by_pair[int(perm[e["pair"]])][e["slot"]]["image_id"]) for e in exs], [True] * len(exs))
    out["none"] = ([dict(e) for e in exs], [False] * len(exs))
    return out


@torch.no_grad()
def run_condition(vlm, store, xs, flags, bs, max_new):
    amp = vlm.device.type == "cuda"
    preds, nlls, t_gen = [], [], 0.0
    for i in range(0, len(xs), bs):
        chunk, fl = xs[i:i + bs], flags[i:i + bs]
        gen_in = [dict(e, target=None) for e in chunk]
        ids, _, att, f, h = vlm.collate(gen_in, store.get, fl, pad_left=True)
        if amp:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast(device_type=vlm.device.type, dtype=torch.float16, enabled=amp):
            preds += vlm.generate(ids, att, f, h, max_new)
        if amp:
            torch.cuda.synchronize()
        t_gen += time.perf_counter() - t0
        ids, lab, att, f, h = vlm.collate(chunk, store.get, fl, label_eot=False)
        with torch.autocast(device_type=vlm.device.type, dtype=torch.float16, enabled=amp):
            nlls += vlm.per_example_nll(ids, lab, att, f, h).tolist()
    return preds, np.array(nlls), t_gen / len(xs)


def score(e, pred):
    pn = normalize(pred)
    em = float(pn == normalize(e["target"]))
    acc = soft_accuracy(pred, e["answers"]) if e["kind"] == "vqa" else em
    return acc, em, pn


def boot_ci(per_pair, B, seed):
    """per_pair: [n_pairs] array (or dict of arrays sharing resampled pair IDs). Returns mean and 95% CI."""
    rng = np.random.default_rng(seed)
    n = len(per_pair)
    idx = rng.integers(0, n, size=(B, n))
    bs = per_pair[idx].mean(1)
    return {"mean": float(per_pair.mean()), "ci_low": float(np.percentile(bs, 2.5)), "ci_high": float(np.percentile(bs, 97.5))}


def evaluate_suite(vlm, store, pairs, suite, cfg, run_dir_res):
    ec = cfg["eval"]
    exs = pair_examples(pairs, "val2014")
    n_pairs = len(pairs)
    perm = derangement(n_pairs, cfg["seed"])
    conds = condition_examples(exs, perm)
    rows, res = [], {}
    raw = {}
    for c in CONDS:
        xs, fl = conds[c]
        preds, nlls, tpa = run_condition(vlm, store, xs, fl, ec["batch_size"], ec["max_new_tokens"])
        raw[c] = {"preds": preds, "nll": nlls, "time_per_answer_s": tpa}
        for e, x, p, nl in zip(exs, xs, preds, nlls):
            acc, em, pn = score(e, p)
            partner = exs[2 * e["pair"] + (1 - e["slot"])]
            follow = soft_accuracy(p, partner["answers"]) if e["kind"] == "vqa" else float(pn == normalize(partner["target"]))
            rows.append({"suite": suite, "condition": c, "pair": e["pair"], "slot": e["slot"], "qtype": e["qtype"],
                         "question": e["question"], "reference": e["target"], "answers": e["answers"],
                         "image_id": e["image_id"], "shown_image": x["show_image"] if fl[0] else None,
                         "prediction": p, "pred_norm": pn, "acc": acc, "em": em, "empty": float(pn == ""),
                         "partner_follow": follow, "ref_nll": float(nl)})
    import pandas as pd

    df = pd.DataFrame(rows)
    B, seed = ec["bootstrap"], cfg["seed"]
    pair_tab = {}
    for c in CONDS:
        d = df[df.condition == c].sort_values(["pair", "slot"])
        acc = d.acc.to_numpy().reshape(n_pairs, 2)
        em = d.em.to_numpy().reshape(n_pairs, 2)
        pn = d.pred_norm.to_numpy().reshape(n_pairs, 2)
        nll = d.ref_nll.to_numpy().reshape(n_pairs, 2)
        pair_tab[c] = {"acc": acc.mean(1), "pair_acc": (em.min(1) == 1).astype(float),
                       "identical": (pn[:, 0] == pn[:, 1]).astype(float),
                       "empty": d["empty"].to_numpy().reshape(n_pairs, 2).mean(1), "nll": nll.mean(1),
                       "follow": d.partner_follow.to_numpy().reshape(n_pairs, 2).mean(1)}
        r = {k: boot_ci(v, B, seed) for k, v in pair_tab[c].items()}
        r["time_per_answer_s"] = raw[c]["time_per_answer_s"]
        r["by_type"] = {qt: {"acc": float(g.acc.mean()), "em": float(g.em.mean()), "nll": float(g.ref_nll.mean()),
                             "n_questions": int(len(g))} for qt, g in d.groupby("qtype")}
        res[c] = r
    # correct-image advantages (same bootstrap pair indices for both conditions via paired per-pair differences)
    adv = {}
    for c in ("complementary", "mismatched", "none"):
        adv[c] = {"acc_advantage": boot_ci(pair_tab["correct"]["acc"] - pair_tab[c]["acc"], B, seed),
                  "nll_advantage": boot_ci(pair_tab[c]["nll"] - pair_tab["correct"]["nll"], B, seed)}
        types = sorted(df.qtype.unique())
        d0 = df[df.condition == "correct"].sort_values(["pair", "slot"])
        d1 = df[df.condition == c].sort_values(["pair", "slot"])
        adv[c]["by_type"] = {qt: {"acc_advantage": float(d0[d0.qtype == qt].acc.mean() - d1[d1.qtype == qt].acc.mean()),
                                  "nll_advantage": float(d1[d1.qtype == qt].ref_nll.mean() - d0[d0.qtype == qt].ref_nll.mean())}
                             for qt in types}
    df.to_json(run_dir_res / f"{suite}_rows.jsonl", orient="records", lines=True, force_ascii=False)
    save_json({"mismatched_pair_map": perm.tolist()}, run_dir_res / f"{suite}_mismatch_map.json")
    return {"conditions": res, "correct_image_advantage": adv, "n_pairs": n_pairs}, df


def qualitative(dfs, k, out):
    """Fixed rule: the first k pairs (saved pair order) of each question type in each test suite."""
    ex = []
    for suite, df in dfs.items():
        for qt, g in df[df.condition == "correct"].groupby("qtype"):
            for pid in sorted(g.pair.unique())[:k]:
                pr = df[df.pair == pid]
                item = {"suite": suite, "qtype": qt, "pair": int(pid), "question": pr.question.iat[0], "slots": []}
                for s in (0, 1):
                    r = pr[pr.slot == s]
                    item["slots"].append({"image_id": int(r.image_id.iat[0]), "reference": r.reference.iat[0],
                                          "answers": r.answers.iat[0],
                                          **{f"pred_{c}": r[r.condition == c].prediction.iat[0] for c in CONDS}})
                ex.append(item)
    save_json({"rule": f"first {k} pairs per question type in each test suite (saved order)", "examples": ex},
              out / "qualitative_pairs.json")
    return ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs/task2.yaml"))
    ap.add_argument("--run", default="vis196")
    ap.add_argument("--weights", default="stage3_weights.pt")
    ap.add_argument("--n_vis", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    n_vis = args.n_vis or (49 if args.run == "vis49" else cfg["n_visual_tokens"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vlm, tok = build_vlm(cfg, device, n_vis)
    w = torch.load(out_root() / "task2/runs" / args.run / args.weights, map_location="cpu")
    vlm.connector.load_state_dict(w["connector"])
    if w["lora"]:
        add_lora(vlm, cfg)
        L.load_adapter_state_dict(vlm.llm, w["lora"])
    vlm.eval()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    store = FeatureStore(n_vis)
    data = load_all()
    out = results_root() / "task2" / args.run / "final_eval"
    out.mkdir(parents=True, exist_ok=True)
    summary, dfs = {"weights": args.weights, **gpu_info()}, {}
    for suite in ("grounding_test", "vqa_test"):
        summary[suite], dfs[suite] = evaluate_suite(vlm, store, data[suite], suite, cfg, out)
        c = summary[suite]["conditions"]
        print(suite, {k: round(v["acc"]["mean"], 3) for k, v in c.items()},
              "adv vs comp:", round(summary[suite]["correct_image_advantage"]["complementary"]["acc_advantage"]["mean"], 3))
    summary["peak_mem_gb"] = peak_mem_gb()["peak_alloc_gb"]
    summary["trainable_params"] = {"connector": sum(p.numel() for p in vlm.connector.parameters()),
                                   "lora": sum(m.A.numel() + m.B.numel() for m in L.lora_modules(vlm.llm).values())}
    save_json(summary, out / "summary.json")
    qualitative(dfs, cfg["eval"]["n_qualitative_per_type"], out)


if __name__ == "__main__":
    main()
