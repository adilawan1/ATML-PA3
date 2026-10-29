"""Shared helpers: seeding, IO, timing, GPU memory, RNG state capture."""
import json
import os
import random
import re
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def out_root() -> Path:
    """Root for large artifacts (checkpoints, caches). Point PA3_OUT at Google Drive on Colab."""
    return Path(os.environ.get("PA3_OUT", REPO_ROOT / "outputs"))


def results_root() -> Path:
    """Small machine-readable results that get committed (PA3_RESULTS overrides, e.g. for smoke tests)."""
    return Path(os.environ.get("PA3_RESULTS", REPO_ROOT / "results"))


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj, path, indent=2):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False, default=_json_default)
    os.replace(tmp, path)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=_json_default) + "\n")


def atomic_torch_save(obj, path):
    """Write to a temp file then rename, so an interrupted save never leaves a truncated checkpoint."""
    import torch

    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def truncate_jsonl(path, key, max_value):
    """Drop log lines with record[key] > max_value (entries written after the checkpoint we resume from)."""
    path = Path(path)
    if not path.exists():
        return 0
    rows = read_jsonl(path)
    keep = [r for r in rows if r.get(key, 0) <= max_value]
    with open(path, "w", encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False, default=_json_default) + "\n")
    return len(rows) - len(keep)


def final_attempt_log(rows, key="update"):
    """Rows of the attempt that produced the final weights: the last occurrence of each key value, in key order.

    A run restarted from scratch before its first checkpoint (or written by an older trainer that did not clear
    the log on a fresh start) repeats the early updates; later attempts always come after earlier ones in the file.
    """
    last = {}
    for r in rows:
        last[r[key]] = r
    return [last[k] for k in sorted(last)]


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_rng_state():
    import torch

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state):
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def gpu_info():
    import torch

    if not torch.cuda.is_available():
        return {"gpu": "cpu"}
    p = torch.cuda.get_device_properties(0)
    return {"gpu": p.name, "gpu_total_mem_gb": round(p.total_memory / 2**30, 2),
            "torch": torch.__version__, "cuda": torch.version.cuda}


def peak_mem_gb(reset=False):
    import torch

    if not torch.cuda.is_available():
        return {"peak_alloc_gb": 0.0, "peak_reserved_gb": 0.0}
    out = {"peak_alloc_gb": torch.cuda.max_memory_allocated() / 2**30,
           "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30}
    if reset:
        torch.cuda.reset_peak_memory_stats()
    return out


def sync():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


@contextmanager
def timed(store: dict, key: str):
    sync()
    t0 = time.perf_counter()
    yield
    sync()
    store[key] = store.get(key, 0.0) + time.perf_counter() - t0


def dir_size_bytes(path) -> int:
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def normalize_text(s: str) -> str:
    """Lowercase, drop non-alphanumerics, collapse whitespace (used for decontamination)."""
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def count_params(params) -> int:
    seen, n = set(), 0
    for p in params:
        if id(p) in seen:
            continue
        seen.add(id(p))
        n += p.numel()
    return n
