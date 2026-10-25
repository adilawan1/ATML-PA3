"""Synthetic Task 2 selections + random SigLIP-shaped features, for smoke-testing (NOT real data)."""
import json
import sys
from pathlib import Path

import numpy as np

res, out = Path(sys.argv[1]), Path(sys.argv[2])
rng = np.random.default_rng(0)
d = res / "task2/data"
d.mkdir(parents=True, exist_ok=True)
train_ids, val_ids = list(range(1, 41)), list(range(1001, 1101))
caps = [{"image_id": i, "caption_id": 10 * i + k, "caption": f"a photo of thing {i % 5} number {k}"} for i in train_ids for k in range(3)]
def pair(kind, q, a, b, ans, answers=None):
    items = []
    for img, x in zip((a, b), ans):
        it = {"image_id": img, "answer": x}
        if answers is not None:
            it.update(question=q, answers=[x] * 9 + ["other"], question_id=img * 7, answer_norm=x, agreement=9)
        items.append(it)
    return {"kind": kind, "question": q, "items": items}
pres = [pair("presence", "Does this image contain a cat?", train_ids[i], train_ids[-i - 1], ["yes", "no"]) for i in range(10)]
cnt = [pair("counting", "How many dogs can you see?", train_ids[i], train_ids[i + 5], [str(1 + i % 4), str(1 + (i + 1) % 4)]) for i in range(10)]
vqa_tr = [pair(t, f"What {t}?", train_ids[i], train_ids[i + 2], ["yes", "no"] if t == "yes/no" else ["2", "3"] if t == "number" else ["red", "blue"], True)
          for i, t in enumerate(["yes/no", "number", "other"] * 3)]
it = iter(val_ids)
def val_pairs(kinds):
    return [pair(k, f"Q {k}?", next(it), next(it), ["yes", "no"] if k in ("presence", "yes/no") else ["1", "2"] if k in ("counting", "number") else ["red", "blue"],
                 True if k not in ("presence", "counting") else None) for k in kinds]
files = {"captions_train": caps, "pool_presence": pres, "pool_counting": cnt, "vqa_train": vqa_tr,
         "vqa_test": val_pairs(["yes/no", "number", "other"] * 2), "vqa_dev": val_pairs(["yes/no", "number", "other"]),
         "grounding_test": val_pairs(["presence", "counting"] * 2), "grounding_dev": val_pairs(["presence", "counting"])}
files["captions_dev"] = [{"image_id": i, "caption_id": i, "caption": "a dev photo"} for i in [next(it) for _ in range(6)]]
for k, v in files.items():
    json.dump(v, open(d / f"{k}.json", "w"))
used_val = sorted({x["image_id"] for k in ("vqa_test", "vqa_dev", "grounding_test", "grounding_dev") for p in files[k] for x in p["items"]} | {c["image_id"] for c in files["captions_dev"]})
json.dump({"train2014": train_ids, "val2014": used_val}, open(d / "images.json", "w"))
fd = out / "task2/features"
fd.mkdir(parents=True, exist_ok=True)
for split, ids in (("train2014", train_ids), ("val2014", used_val)):
    np.save(fd / f"{split}.npy", rng.normal(0, 1, (len(ids), 196, 768)).astype(np.float16))
    json.dump({str(i): r for r, i in enumerate(ids)}, open(fd / f"{split}_index.json", "w"))
print("ok")
