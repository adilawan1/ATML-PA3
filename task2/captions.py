"""Generate dev-set captions after each training stage (Task 2, RQ1: "describe images").

Training only tracked caption *loss*; this script produces the caption text itself. For each saved stage
checkpoint (stage1/2/3_weights.pt) it captions every dev-caption image (96 val2014 images, never used in
training) with the training prompt "Write a short caption for this image.", greedy decoding, thinking
disabled, 64-token cap (the caption length cap used in training). It also generates once with no image as a
language-prior control (the no-image prompt is identical for every image, so one generation suffices).

Selection rule for examples quoted in the report: the first 8 dev images in saved order
(results/task2/data/captions_dev.json); nothing is chosen after seeing outputs.

  python -m task2.captions [--run vis196] [--n_show 8] [--bs 8]
Writes results/task2/<run>/captions/dev_captions.json
"""
import argparse
import time

import torch

from common import lora as L
from common.utils import REPO_ROOT, gpu_info, load_config, out_root, results_root, save_json
from task2.data import caption_examples
from task2.vlm import FeatureStore, add_lora, build_vlm


@torch.no_grad()
def caption_all(vlm, store, exs, bs, max_new):
    amp = vlm.device.type == "cuda"
    out = []
    for i in range(0, len(exs), bs):
        chunk = [dict(e, target=None) for e in exs[i:i + bs]]
        ids, _, att, f, h = vlm.collate(chunk, store.get, [True] * len(chunk), pad_left=True)
        with torch.autocast(device_type=vlm.device.type, dtype=torch.float16, enabled=amp):
            out += vlm.generate(ids, att, f, h, max_new)
    return out


@torch.no_grad()
def caption_no_image(vlm, max_new):
    amp = vlm.device.type == "cuda"
    ex = {"kind": "caption", "target": None}
    ids, _, att, f, h = vlm.collate([ex], None, [False], pad_left=True)
    with torch.autocast(device_type=vlm.device.type, dtype=torch.float16, enabled=amp):
        return vlm.generate(ids, att, f, h, max_new)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs/task2.yaml"))
    ap.add_argument("--run", default="vis196")
    ap.add_argument("--n_show", type=int, default=8)
    ap.add_argument("--bs", type=int, default=8)
    args = ap.parse_args()
    cfg = load_config(args.config)
    n_vis = 49 if args.run == "vis49" else cfg["n_visual_tokens"]
    max_new = cfg["train"]["caps"]["caption"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vlm, tok = build_vlm(cfg, device, n_vis)
    vlm.eval()
    store = FeatureStore(n_vis)
    exs = caption_examples("captions_dev", "val2014")
    rd = out_root() / "task2/runs" / args.run

    res = {"prompt": cfg["prompts"]["caption"], "decoding": "greedy", "max_new_tokens": max_new,
           "n_images": len(exs), "selection_rule": f"first {args.n_show} dev images in saved order",
           "device": device, **gpu_info(), "stages": {}}
    lora_added = False
    for k in (1, 2, 3):
        w = torch.load(rd / f"stage{k}_weights.pt", map_location="cpu")
        vlm.connector.load_state_dict(w["connector"])
        if w["lora"]:
            if not lora_added:
                add_lora(vlm, cfg)
                lora_added = True
            L.load_adapter_state_dict(vlm.llm, w["lora"])
        vlm.eval()
        t0 = time.perf_counter()
        caps = caption_all(vlm, store, exs, args.bs, max_new)
        dt = time.perf_counter() - t0
        none = caption_no_image(vlm, max_new)
        n_unique = len(set(c.strip().lower() for c in caps))
        res["stages"][f"stage{k}"] = {
            "lora": bool(w["lora"]),
            "seconds_per_caption": dt / len(exs),
            "n_unique_captions": n_unique,
            "n_empty": sum(not c.strip() for c in caps),
            "mean_words": sum(len(c.split()) for c in caps) / len(caps),
            "no_image_caption": none,
            "captions": [{"image_id": e["image_id"], "reference": e["target"], "generated": c}
                         for e, c in zip(exs, caps)],
        }
        print(f"stage{k}: {n_unique}/{len(exs)} unique, no-image: {none!r}", flush=True)
        for e, c in list(zip(exs, caps))[: args.n_show]:
            print(f"  {e['image_id']}: {c!r}   | ref: {e['target']!r}")
    out = results_root() / "task2" / args.run / "captions"
    out.mkdir(parents=True, exist_ok=True)
    save_json(res, out / "dev_captions.json")
    print("wrote", out / "dev_captions.json")


if __name__ == "__main__":
    main()
