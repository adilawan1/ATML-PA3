"""Download the selected COCO images and cache frozen SigLIP patch features (FP16, 196 x 768 per image).

Features are SigLIP's `last_hidden_state`: the 196 final patch tokens in 14 x 14 row-major order with the
final post-layernorm applied exactly once (verified numerically below; the MAP pooling head is not used).
Preprocessing is the deterministic SigLIP image processor (resize 224, rescale, normalize).

  python -m task2.cache_features
Outputs: $PA3_OUT/task2/features/{train2014,val2014}.npy (+ _index.json); results/task2/feature_cache.json
"""
import argparse
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

from common.utils import REPO_ROOT, gpu_info, load_config, load_json, out_root, results_root, save_json
from task2.prepare_data import coco_root


def img_path(root, split, iid):
    return root / split / f"COCO_{split}_{iid:012d}.jpg"


def download(root, split, ids, workers=32):
    (root / split).mkdir(parents=True, exist_ok=True)
    todo = [i for i in ids if not img_path(root, split, i).exists()]

    def get(i):
        p = img_path(root, split, i)
        for attempt in range(5):
            try:
                urllib.request.urlretrieve(f"http://images.cocodataset.org/{split}/{p.name}", p.with_suffix(".part"))
                p.with_suffix(".part").rename(p)
                return
            except Exception:
                time.sleep(1 + attempt)
        raise RuntimeError(f"failed to download {p.name}")

    t0 = time.perf_counter()
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(get, todo))
    return {"downloaded": len(todo), "download_time_s": time.perf_counter() - t0}


class Imgs(torch.utils.data.Dataset):
    def __init__(self, paths, proc):
        self.paths, self.proc = paths, proc

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert("RGB")
        return self.proc(images=im, return_tensors="pt")["pixel_values"][0]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    from transformers import AutoImageProcessor, SiglipVisionModel

    cfg = load_config(REPO_ROOT / "configs/task2.yaml")
    root = coco_root(cfg)
    images = load_json(results_root() / "task2/data/images.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rep = {**gpu_info(), "vision": cfg["vision"]}
    for split, ids in images.items():
        rep[f"{split}_download"] = download(root, split, ids)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    proc = AutoImageProcessor.from_pretrained(cfg["vision"])
    model = SiglipVisionModel.from_pretrained(cfg["vision"], dtype=torch.float32).to(device).eval()
    rep["preprocessing"] = {k: str(v) for k, v in proc.to_dict().items() if k in
                            ("size", "resample", "do_rescale", "rescale_factor", "do_normalize", "image_mean", "image_std")}
    fd = out_root() / "task2/features"
    fd.mkdir(parents=True, exist_ok=True)
    checked = False
    for split, ids in images.items():
        t0 = time.perf_counter()
        paths = [img_path(root, split, i) for i in ids]
        mm = np.lib.format.open_memmap(fd / f"{split}.npy", mode="w+", dtype=np.float16, shape=(len(ids), 196, 768))
        dl = torch.utils.data.DataLoader(Imgs(paths, proc), batch_size=args.batch, num_workers=args.workers)
        row = 0
        for px in dl:
            out = model(pixel_values=px.to(device), output_hidden_states=not checked)
            feats = out.last_hidden_state
            if not checked:
                # last_hidden_state == post_layernorm(final encoder output): LN is applied exactly once
                post_ln = next(m for n, m in model.named_modules() if n.endswith("post_layernorm"))
                ref = post_ln(out.hidden_states[-1])
                rep["hidden_states_last_equals_output"] = bool(torch.allclose(out.hidden_states[-1], feats))
                rep["post_layernorm_check_max_abs_diff"] = (feats - ref).abs().max().item()
                assert rep["post_layernorm_check_max_abs_diff"] < 1e-4
                assert feats.shape[1:] == (196, 768), feats.shape
                checked = True
            mm[row:row + len(feats)] = feats.half().cpu().numpy()
            row += len(feats)
        mm.flush()
        save_json({str(i): r for r, i in enumerate(ids)}, fd / f"{split}_index.json", indent=None)
        rep[f"{split}_cache"] = {"n_images": len(ids), "time_s": time.perf_counter() - t0,
                                 "bytes": (fd / f"{split}.npy").stat().st_size}
        print(split, rep[f"{split}_cache"], flush=True)
    rep["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    save_json(rep, results_root() / "task2/feature_cache.json")


if __name__ == "__main__":
    main()
