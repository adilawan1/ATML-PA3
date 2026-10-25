"""Select all Task 2 data with seed 6304 and save every ID (no images are read here).

Downloads COCO 2014 caption/instance annotations and VQAv2 questions/annotations/complementary pairs
(unless already present), then builds:
  - COCO captions: 6,000 train2014 images x first 3 captions; 96 val2014 dev images x first caption
  - VQAv2 complementary pairs (both images >= 8/10 agreement on a normalized consensus answer, consensus
    answers differ, same answer type): train 640/256/640, dev 12/8/12, test 36/24/36
  - Annotation-derived grounding: reusable train pools (presence, counting) over the cached train images;
    dev 24 pairs, test 64 pairs from val2014
All dev/test sets are image-disjoint from one another (and from training, which uses train2014 only).

  python -m task2.prepare_data
Outputs: results/task2/data/*.json  and  results/task2/data/images.json (every image to download/cache)
"""
import json
import os
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from common.utils import REPO_ROOT, load_config, results_root, save_json
from task2.vqa_norm import normalize

COCO_ANN = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
VQA_BASE = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/"
VQA_FILES = ["v2_Questions_Train_mscoco.zip", "v2_Questions_Val_mscoco.zip", "v2_Annotations_Train_mscoco.zip",
             "v2_Annotations_Val_mscoco.zip", "v2_Complementary_Pairs_Train_mscoco.zip",
             "v2_Complementary_Pairs_Val_mscoco.zip"]
PLURAL = {"person": "people", "mouse": "mice", "sheep": "sheep", "skis": "skis", "scissors": "scissors",
          "bus": "buses", "couch": "couches", "bench": "benches", "sandwich": "sandwiches", "knife": "knives",
          "wine glass": "wine glasses", "glass": "glasses", "skateboard": "skateboards", "toothbrush": "toothbrushes",
          "dining table": "dining tables", "hair drier": "hair driers", "potted plant": "potted plants",
          "teddy bear": "teddy bears", "fire hydrant": "fire hydrants", "box": "boxes"}


def coco_root(cfg):
    return Path(os.environ.get("PA3_COCO", REPO_ROOT / cfg["data"]["coco_root"]))


def fetch(url, dst_dir):
    dst_dir.mkdir(parents=True, exist_ok=True)
    z = dst_dir / url.rsplit("/", 1)[1]
    if not z.exists():
        print("downloading", url, flush=True)
        urllib.request.urlretrieve(url, z)
    with zipfile.ZipFile(z) as f:
        names = f.namelist()
        if not all((dst_dir / n).exists() for n in names):
            f.extractall(dst_dir)


def download_annotations(root):
    fetch(COCO_ANN, root)
    for f in VQA_FILES:
        fetch(VQA_BASE + f, root / "vqa")


def plural(name):
    return PLURAL.get(name, name + "s")


def article(name):
    return "an" if name[0] in "aeiou" else "a"


# ----------------------------------------------------------------------------- VQAv2
def vqa_candidates(root, split, min_agree):
    s = {"train": "train2014", "val": "val2014"}[split]
    qs = {q["question_id"]: q for q in json.load(open(root / f"vqa/v2_OpenEnded_mscoco_{s}_questions.json"))["questions"]}
    anns = {a["question_id"]: a for a in json.load(open(root / f"vqa/v2_mscoco_{s}_annotations.json"))["annotations"]}
    pairs = json.load(open(root / f"vqa/v2_mscoco_{s}_complementary_pairs.json"))

    def consensus(qid):
        a = anns[qid]
        raw = [x["answer"] for x in a["answers"]]
        norm = [normalize(x) for x in raw]
        top, n = Counter(norm).most_common(1)[0]
        if n < min_agree:
            return None
        # canonical target: most frequent original string within the consensus group (ties: first occurrence)
        grp = [r for r, m in zip(raw, norm) if m == top]
        canon = Counter(grp).most_common(1)[0][0]
        return {"question_id": qid, "image_id": a["image_id"], "question": qs[qid]["question"], "answer": canon,
                "answer_norm": top, "answers": raw, "answer_type": a["answer_type"], "agreement": n}

    out = defaultdict(list)
    for q1, q2 in pairs:
        c1, c2 = consensus(q1), consensus(q2)
        if not c1 or not c2 or c1["answer_norm"] == c2["answer_norm"] or c1["image_id"] == c2["image_id"]:
            continue
        if c1["answer_type"] != c2["answer_type"]:
            continue
        out[c1["answer_type"]].append([c1, c2])
    return out


# ----------------------------------------------------------------------------- COCO instances
def instance_index(root, split, min_frac, max_count):
    d = json.load(open(root / f"annotations/instances_{split}2014.json"))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    img = {i["id"]: i for i in d["images"]}
    per = defaultdict(lambda: defaultdict(list))
    for a in d["annotations"]:
        per[a["image_id"]][a["category_id"]].append(a)
    clean_count, positive, present = defaultdict(dict), defaultdict(set), defaultdict(set)
    for iid, bycat in per.items():
        area = img[iid]["width"] * img[iid]["height"]
        for c, insts in bycat.items():
            present[c].add(iid)
            ok = all(not x["iscrowd"] and x["area"] >= min_frac * area for x in insts)
            if any(not x["iscrowd"] and x["area"] >= min_frac * area for x in insts):
                positive[c].add(iid)
            if ok and 1 <= len(insts) <= max_count:
                clean_count[c][iid] = len(insts)
    return cats, set(img), clean_count, positive, present


def presence_pair(rng, c, cats, pos_imgs, neg_imgs):
    p = pos_imgs[rng.integers(len(pos_imgs))]
    n = neg_imgs[rng.integers(len(neg_imgs))]
    q = f"Does this image contain {article(cats[c])} {cats[c]}?"
    return {"kind": "presence", "category": cats[c], "question": q,
            "items": [{"image_id": int(p), "answer": "yes"}, {"image_id": int(n), "answer": "no"}]}


def counting_pair(rng, c, cats, by_count):
    levels = sorted(k for k, v in by_count.items() if v)  # count levels with at least one image
    a, b = rng.choice(levels, size=2, replace=False)
    ia = by_count[a][rng.integers(len(by_count[a]))]
    ib = by_count[b][rng.integers(len(by_count[b]))]
    q = f"How many {plural(cats[c])} can you see?"
    return {"kind": "counting", "category": cats[c], "question": q,
            "items": [{"image_id": int(ia), "answer": str(a)}, {"image_id": int(ib), "answer": str(b)}]}


class GroundingIndex:
    """Per-category image lists restricted to `allowed` images, precomputed once."""

    def __init__(self, cats, allowed, clean_count, positive, present, max_count):
        self.cats = cats
        self.pos = {c: sorted(positive[c] & allowed) for c in cats}
        self.neg = {c: sorted(allowed - present[c]) for c in cats}
        self.by = {c: {k: sorted(i for i, kk in clean_count[c].items() if kk == k and i in allowed)
                       for k in range(1, max_count + 1)} for c in cats}
        self.presence_cats = sorted(c for c in cats if self.pos[c] and self.neg[c])
        self.counting_cats = sorted(c for c in cats if sum(bool(v) for v in self.by[c].values()) >= 2)


def grounding_set(rng, kind, n, gi, used=None, unique=True):
    """Sample n pairs; category uniform over eligible categories. With unique=True no image is reused across
    the returned pairs or with `used` (rejection sampling; `used` is updated)."""
    used = set() if used is None else used
    out, tries = [], 0
    while len(out) < n:
        tries += 1
        assert tries < 1000 * n + 10000, f"could not sample {n} {kind} pairs"
        if kind == "presence":
            c = gi.presence_cats[rng.integers(len(gi.presence_cats))]
            pair = presence_pair(rng, c, gi.cats, gi.pos[c], gi.neg[c])
        else:
            c = gi.counting_cats[rng.integers(len(gi.counting_cats))]
            pair = counting_pair(rng, c, gi.cats, gi.by[c])
        ids = [it["image_id"] for it in pair["items"]]
        if ids[0] == ids[1]:
            continue
        if unique:
            if set(ids) & used:
                continue
            used.update(ids)
        out.append(pair)
    return out


def take_vqa(rng, cands, counts, used, unique):
    out = []
    for t, n in counts.items():
        pool = cands[t]
        order = rng.permutation(len(pool))
        k = 0
        for i in order:
            if k == n:
                break
            p = pool[i]
            ids = {p[0]["image_id"], p[1]["image_id"]}
            if unique and ids & used:
                continue
            if unique:
                used.update(ids)
            out.append({"kind": t, "question": p[0]["question"], "items": [
                {k2: x[k2] for k2 in ("question_id", "image_id", "question", "answer", "answers", "answer_norm", "agreement")}
                for x in p]})
            k += 1
        assert k == n, f"only {k}/{n} {t} pairs available"
    return out


def main():
    cfg = load_config(REPO_ROOT / "configs/task2.yaml")
    dc = cfg["data"]
    root = coco_root(cfg)
    download_annotations(root)
    rng = np.random.default_rng(cfg["seed"])
    od = results_root() / "task2/data"

    # ---- training (train2014)
    vqa_tr = vqa_candidates(root, "train", dc["vqa_min_agreement"])
    vqa_train = take_vqa(rng, vqa_tr, dc["vqa_pairs"]["train"], set(), unique=False)
    caps = json.load(open(root / "annotations/captions_train2014.json"))
    by_img = defaultdict(list)
    for a in sorted(caps["annotations"], key=lambda a: a["id"]):
        by_img[a["image_id"]].append(a)
    cap_imgs = sorted(int(i) for i in rng.choice(sorted(by_img), dc["captions_train_images"], replace=False))
    cap_train = [{"image_id": i, "caption_id": a["id"], "caption": a["caption"].strip()}
                 for i in cap_imgs for a in by_img[i][:dc["captions_per_train_image"]]]
    train_imgs = set(cap_imgs) | {it["image_id"] for p in vqa_train for it in p["items"]}
    cats, _, cc, pos, pres = instance_index(root, "train", dc["min_instance_area_frac"], dc["max_count"])
    gi_train = GroundingIndex(cats, train_imgs, cc, pos, pres, dc["max_count"])
    pools = {k: grounding_set(rng, k, n, gi_train, unique=False) for k, n in dc["grounding_pool_size"].items()}

    # ---- development / final test (val2014), mutually image-disjoint
    used = set()
    vqa_va = vqa_candidates(root, "val", dc["vqa_min_agreement"])
    vqa_test = take_vqa(rng, vqa_va, dc["vqa_pairs"]["test"], used, unique=True)
    vqa_dev = take_vqa(rng, vqa_va, dc["vqa_pairs"]["dev"], used, unique=True)
    vcats, vimgs, vcc, vpos, vpres = instance_index(root, "val", dc["min_instance_area_frac"], dc["max_count"])
    gi_val = GroundingIndex(vcats, vimgs, vcc, vpos, vpres, dc["max_count"])
    g_test, g_dev = [], []
    for k, n in dc["grounding_pairs"]["test"].items():
        g_test += grounding_set(rng, k, n, gi_val, used=used)
    for k, n in dc["grounding_pairs"]["dev"].items():
        g_dev += grounding_set(rng, k, n, gi_val, used=used)
    vcaps = json.load(open(root / "annotations/captions_val2014.json"))
    vby = defaultdict(list)
    for a in sorted(vcaps["annotations"], key=lambda a: a["id"]):
        vby[a["image_id"]].append(a)
    free = sorted(set(vby) - used)
    dev_cap_imgs = sorted(int(i) for i in rng.choice(free, dc["captions_dev_images"], replace=False))
    used.update(dev_cap_imgs)
    cap_dev = [{"image_id": i, "caption_id": vby[i][0]["id"], "caption": vby[i][0]["caption"].strip()} for i in dev_cap_imgs]

    # ---- isolation checks
    eval_sets = {"vqa_test": vqa_test, "vqa_dev": vqa_dev, "grounding_test": g_test, "grounding_dev": g_dev}
    img_sets = {k: [it["image_id"] for p in v for it in p["items"]] for k, v in eval_sets.items()}
    img_sets["caption_dev"] = dev_cap_imgs
    for k, v in img_sets.items():
        assert len(v) == len(set(v)), f"duplicate images within {k}"
    names = list(img_sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert not set(img_sets[names[i]]) & set(img_sets[names[j]]), f"{names[i]} overlaps {names[j]}"
    train_all = train_imgs | {it["image_id"] for v in pools.values() for p in v for it in p["items"]}
    assert train_all <= train_imgs

    for name, obj in [("captions_train", cap_train), ("captions_dev", cap_dev), ("vqa_train", vqa_train),
                      ("vqa_dev", vqa_dev), ("vqa_test", vqa_test), ("grounding_dev", g_dev), ("grounding_test", g_test),
                      ("pool_presence", pools["presence"]), ("pool_counting", pools["counting"])]:
        save_json(obj, od / f"{name}.json", indent=None)
    images = {"train2014": sorted(int(i) for i in train_imgs), "val2014": sorted(int(i) for i in used)}
    save_json(images, od / "images.json", indent=None)
    stats = {"n_train_images": len(images["train2014"]), "n_val_images": len(images["val2014"]),
             "vqa_train_candidates": {k: len(v) for k, v in vqa_tr.items()},
             "vqa_val_candidates": {k: len(v) for k, v in vqa_va.items()},
             "captions_train": len(cap_train), "captions_dev": len(cap_dev),
             "pool_sizes": {k: len(v) for k, v in pools.items()},
             "pool_unique_pairs": {k: len({(p["category"], *[i["image_id"] for i in p["items"]]) for p in v})
                                   for k, v in pools.items()},
             "pool_answer_counts": {k: Counter(i["answer"] for p in v for i in p["items"]) for k, v in pools.items()},
             "eval_answer_counts": {k: Counter(i["answer"] for p in v for i in p["items"]) for k, v in eval_sets.items()
                                    if k.startswith("grounding")}}
    save_json(stats, od / "stats.json")
    print(json.dumps(stats, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
