"""Turn the saved selections (results/task2/data/*.json) into training / evaluation examples."""
from common.utils import load_json, results_root

GROUNDING = ("presence", "counting")


def _d(name):
    return load_json(results_root() / "task2/data" / f"{name}.json")


def caption_examples(name, split):
    return [{"kind": "caption", "target": c["caption"], "image_id": c["image_id"], "show_split": split,
             "show_image": c["image_id"], "caption_id": c["caption_id"]} for c in _d(name)]


def pair_examples(pairs, split):
    """Each pair -> two examples (same question, different image and reference answer)."""
    out = []
    for pi, p in enumerate(pairs):
        for j, it in enumerate(p["items"]):
            kind = p["kind"] if p["kind"] in GROUNDING else "vqa"
            out.append({"kind": kind, "qtype": p["kind"], "question": it.get("question", p["question"]),
                        "target": it["answer"], "answers": it.get("answers"), "image_id": it["image_id"],
                        "show_split": split, "show_image": it["image_id"], "pair": pi, "slot": j,
                        "question_id": it.get("question_id")})
    return out


def load_all():
    return {
        "captions_train": caption_examples("captions_train", "train2014"),
        "captions_dev": caption_examples("captions_dev", "val2014"),
        "pool_presence": _d("pool_presence"), "pool_counting": _d("pool_counting"),
        "vqa_train": _d("vqa_train"),
        "grounding_dev": _d("grounding_dev"), "vqa_dev": _d("vqa_dev"),
        "grounding_test": _d("grounding_test"), "vqa_test": _d("vqa_test"),
    }
