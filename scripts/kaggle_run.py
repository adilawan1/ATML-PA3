"""Kaggle driver: runs one session of PA3 jobs on a 2 x T4 machine, two GPU lanes in parallel.

Kaggle keeps only /kaggle/working (20 GB) after a "Save Version" run, so training writes to scratch disk
($PA3_OUT) and only what later steps need is copied to /kaggle/working/outputs: model weights kept for
evaluation, adapters, OPLoRA bases, logs and summaries (no optimizer state, which is only for resuming).
Small results go to $PA3_RESULTS (/kaggle/working/results); the notebook commits them to GitHub.

Sessions (each fits well inside Kaggle's 12-hour limit):
  t1-check   memory check of all five variants + baseline evaluation (validates lm-eval early)
  t1-fullft  GPU0: full_ft -> eval full_ft_mid, full_ft        GPU1: full_ft_l2sp -> eval full_ft_l2sp
  t1-lora    GPU0: lora4 -> eval lora4                          GPU1: lora16 -> eval lora16
  t1-final   GPU0: oplora16 -> eval oplora16    GPU1: any condition still missing (from attached outputs)
             then diagnostics, qualitative collection, tables and figures. Attach the outputs of the
             t1-fullft and t1-lora notebooks as inputs first.
  t2         Task 2: download + cache features, train vis196 (GPU0) and the optional vis49 (GPU1), evaluate,
             analyze.

  python scripts/kaggle_run.py t1-fullft [--dry_run]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from common.utils import load_config, load_json, out_root, results_root  # noqa: E402

START = time.time()
WORK = Path(os.environ.get("PA3_KEEP", "/kaggle/working/outputs"))  # persisted copy of large artifacts
EVAL_DEADLINE_H = 10.0      # do not start a new evaluation after this many hours (Kaggle stops at 12 h)
CHUNK = 1024                # LM-head block size used when the memory check says the default does not fit
LOG_DIR = Path(os.environ.get("PA3_LOGS", "/kaggle/working/logs"))
DRY = False


def hours():
    return (time.time() - START) / 3600


def run(cmd, log, env=None):
    line = " ".join(map(str, cmd))
    with open(log, "a") as f:
        f.write(f"\n[{time.strftime('%H:%M:%S')} +{hours():.2f}h] $ {line}\n")
        f.flush()
        if DRY:
            return
        r = subprocess.run(list(map(str, cmd)), cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"failed ({r.returncode}): {line}  -- see {log}")


def lane_env(gpu):
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


def n_gpus():
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
        return sum(1 for l in out.splitlines() if l.startswith("GPU"))
    except FileNotFoundError:
        return 0


LANE = None  # --lane: run only this lane index (to split a session in two)
NEED_GB = {"t1-fullft": 36, "t1-check": 8, "t1-lora": 8, "t1-final": 8, "t2": 15}  # transient scratch disk per session
NEED_GB_ONE_FULL_LANE = 18  # one Full FT run holds up to 3 checkpoints at once (incl. FP32 optimizer state)


def check_disk(session):
    """Fail in the first minute rather than hours in: Full FT checkpoints are large."""
    need = NEED_GB_ONE_FULL_LANE if session == "t1-fullft" and LANE is not None else NEED_GB[session]
    out_root().mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out_root()).free / 2**30
    print(f"scratch disk free: {free:.0f} GB (need ~{need} GB)", flush=True)
    if free < need and not DRY:
        raise SystemExit(f"STOP: not enough scratch disk at {out_root()} ({free:.0f} GB free, ~{need} GB needed). "
                         "For t1-fullft, run it as two sessions: --lane 0, then --lane 1.")


def run_lanes(lanes):
    """lanes: list of (name, [callables taking (log, env)]). Two GPUs -> parallel; otherwise sequential on GPU 0."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if LANE is not None:
        lanes = [lanes[LANE]] if LANE < len(lanes) else []
    g = n_gpus()
    parallel = g >= 2 and len(lanes) > 1
    print(f"{g} GPU(s); lanes run {'in parallel' if parallel else 'one after another'}", flush=True)
    errors = {}

    def go(i, name, steps):
        log = LOG_DIR / f"{name}.log"
        env = lane_env(i if parallel else (0 if g else None))
        try:
            for step in steps:
                step(log, env)
        except Exception as e:  # keep the other lane running; report at the end
            errors[name] = repr(e)

    threads = [threading.Thread(target=go, args=(i, n, s)) for i, (n, s) in enumerate(lanes)]
    if parallel:
        for t in threads:
            t.start()
        while any(t.is_alive() for t in threads):
            for t in threads:
                t.join(timeout=600)
            alive = [n for (n, _), t in zip(lanes, threads) if t.is_alive()]
            if alive:
                print(f"+{hours():.2f}h still running: {alive} | " + " | ".join(last_line(LOG_DIR / f'{n}.log') for n in alive), flush=True)
    else:
        for t in threads:
            t.start()
            t.join()
    for name, err in errors.items():
        print(f"LANE {name} FAILED: {err}", flush=True)
    return errors


def last_line(p):
    try:
        lines = [l for l in p.read_text(errors="ignore").replace("\r", "\n").splitlines() if l.strip()]
        return lines[-1][-160:] if lines else ""
    except FileNotFoundError:
        return ""


# ----------------------------------------------------------------------------- Task 1 steps
CFG1 = load_config(REPO / "configs/task1.yaml")


def prepare_task1():
    """Recreate the training tokens (deterministic) and confirm the selection matches the committed one."""
    committed = load_json(REPO / "results/task1/data_selection.json")
    if not (out_root() / "task1/data/train_tokens.npz").exists():
        run([sys.executable, "-m", "task1.prepare_data"], LOG_DIR / "prepare.log", lane_env(None))
    if DRY:
        return
    new = load_json(results_root() / "task1/data_selection.json")
    if new["source_indices"] != committed["source_indices"]:
        raise SystemExit("STOP: the selected MetaMathQA rows differ from the committed selection")
    if new["token_hash_sha256"] != committed["token_hash_sha256"]:
        print("WARNING: same rows, but the token hash differs from the committed one (tokenizer version?). "
              "Recorded in results/task1/kaggle/token_hash_note.json", flush=True)
        save_note({"committed": committed["token_hash_sha256"], "kaggle": new["token_hash_sha256"]},
                  "token_hash_note.json")
    print("training data matches the committed selection", flush=True)


def save_note(obj, name):
    p = results_root() / "task1/kaggle" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2))


def memcheck(variant, log, env):
    """Pick the cheapest memory settings that fit; returns extra train.py arguments."""
    method = CFG1["variants"][variant]["method"]
    l2sp = float(CFG1["variants"][variant].get("l2sp", 0)) > 0
    options = [(0, "gpu"), (CHUNK, "gpu")] + ([(CHUNK, "cpu")] if l2sp else [])
    d = results_root() / "task1/kaggle"
    d.mkdir(parents=True, exist_ok=True)
    tried = []
    for chunk, t0 in options:
        out = d / f"memcheck_{variant}_c{chunk}_{t0}.json"
        run([sys.executable, "-m", "task1.memcheck", "--variant", variant, "--loss_chunk", chunk,
             "--theta0_device", t0, "--out", out], log, env)
        if DRY:
            return []
        rec = json.loads(out.read_text())
        tried.append(rec)
        if rec["headroom_ok"]:
            save_note({"variant": variant, "method": method, "chosen": {"loss_chunk": chunk, "theta0_device": t0},
                       "tried": [{k: r.get(k) for k in ("loss_chunk", "theta0_device", "ok", "peak_reserved_gb",
                                                        "total_gb", "gpu")} for r in tried]},
                      f"memory_{variant}.json")
            with open(log, "a") as f:
                f.write(f"memory setting for {variant}: loss_chunk={chunk} theta0_device={t0}\n")
            return ["--loss_chunk", chunk, "--theta0_device", t0]
    raise RuntimeError(f"{variant} does not fit in GPU memory with any setting: {tried}")


def run_complete(variant):
    p = out_root() / "task1/runs" / variant / "summary.json"
    return p.exists() and load_json(p).get("completed_updates") == CFG1["train"]["updates"]


def train_step(variant):
    def step(log, env):
        if run_complete(variant):
            print(f"{variant}: already trained (attached or earlier), skipped", flush=True)
            return
        extra = memcheck(variant, log, env)
        run([sys.executable, "-m", "task1.train", "--variant", variant, *extra], log, env)
        keep_run(variant)
    return step


def keep_run(variant):
    """Copy a finished run to /kaggle/working without optimizer state (only needed for resuming)."""
    src = out_root() / "task1/runs" / variant
    dst = WORK / "task1/runs" / variant
    if DRY or not src.exists() or src.resolve() == dst.resolve():
        return
    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("state.pt", "*.tmp"))


def evaluated(cond):
    r = results_root() / "task1"
    return (r / "eval" / cond / "scores.json").exists() and (r / "qualitative" / f"{cond}.json").exists()


def eval_step(cond):
    def step(log, env):
        if evaluated(cond):
            print(f"{cond}: already evaluated, skipped", flush=True)
            return
        if hours() > EVAL_DEADLINE_H:
            print(f"{cond}: past {EVAL_DEADLINE_H} h, left for the t1-final session", flush=True)
            return
        merged = Path(os.environ.get("PA3_SCRATCH", "/kaggle/temp")) / "merged"
        py = sys.executable
        run([py, "-m", "task1.export_models", "--condition", cond, "--out_dir", merged], log, env)
        run([py, "-m", "task1.evaluate", "--condition", cond, "--model_dir", merged / cond], log, env)
        run([py, "-m", "task1.qualitative", "--condition", cond, "--model_dir", merged / cond], log, env)
        shutil.rmtree(merged / cond, ignore_errors=True)
        print(f"{cond}: evaluated (+{hours():.2f}h)", flush=True)
    return step


def link_attached_runs():
    """Make runs saved by earlier sessions (attached as notebook inputs) visible under $PA3_OUT."""
    found = {}
    for d in sorted(Path("/kaggle/input").glob("**/outputs/task1/runs/*")) if Path("/kaggle/input").exists() else []:
        if d.is_dir():
            found.setdefault(d.name, d)
    for name, d in found.items():
        tgt = out_root() / "task1/runs" / name
        if not tgt.exists():
            tgt.parent.mkdir(parents=True, exist_ok=True)
            tgt.symlink_to(d, target_is_directory=True)
            print(f"linked attached run {name} <- {d}", flush=True)
    return found


def merge_attached_results():
    """Copy results from attached earlier sessions that are not here yet (in case they were not pushed)."""
    root = Path("/kaggle/input")
    if not root.exists():
        return
    for d in sorted(root.glob("**/results/task1")):
        for f in d.rglob("*"):
            if f.is_file():
                tgt = results_root() / "task1" / f.relative_to(d)
                if not tgt.exists():
                    tgt.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, tgt)


def has_checkpoint(cond):
    c = next(c for c in CFG1["eval"]["conditions"] if c["name"] == cond)
    if c["variant"] is None:
        return True
    ck = out_root() / "task1/runs" / c["variant"] / f"ckpt_u{c['update']}"
    return (ck / "model").exists() or (ck / "adapter_effective.pt").exists()


def remaining_evals_step(exclude):
    def step(log, env):
        for c in (c["name"] for c in CFG1["eval"]["conditions"]):
            if c in exclude or evaluated(c):
                continue
            if has_checkpoint(c):
                eval_step(c)(log, env)
            else:
                print(f"{c}: no checkpoint attached or trained; cannot evaluate yet", flush=True)
    return step


def finalize_task1():
    log, env = LOG_DIR / "final.log", lane_env(0 if n_gpus() else None)
    missing = [c["name"] for c in CFG1["eval"]["conditions"] if not evaluated(c["name"])]
    if missing and not DRY:
        print(f"WARNING: not evaluated yet: {missing}; tables will be partial", flush=True)
    run([sys.executable, "-m", "task1.diagnostics"], log, env)
    run([sys.executable, "-m", "task1.qualitative", "--collect"], log, env)
    run([sys.executable, "-m", "task1.analyze"], log, env)


# ----------------------------------------------------------------------------- Task 2 steps
def t2_train(run_name, n_vis):
    def step(log, env):
        run([sys.executable, "-m", "task2.train", "--n_vis", n_vis], log, env)
        run([sys.executable, "-m", "task2.evaluate", "--run", run_name], log, env)
        src, dst = out_root() / "task2/runs" / run_name, WORK / "task2/runs" / run_name
        if not DRY and src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("ckpt_u*", "*.tmp"))
    return step


# ----------------------------------------------------------------------------- sessions
def main():
    global DRY, LANE
    ap = argparse.ArgumentParser()
    ap.add_argument("session", choices=["t1-check", "t1-fullft", "t1-lora", "t1-final", "t2"])
    ap.add_argument("--no_vis49", action="store_true", help="t2: skip the optional 49-token run")
    ap.add_argument("--dry_run", action="store_true", help="print the plan to the logs without running anything")
    ap.add_argument("--lane", type=int, choices=[0, 1], default=None, help="run only one GPU lane of the session")
    args = ap.parse_args()
    DRY = args.dry_run
    LANE = args.lane
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"session {args.session} | PA3_OUT={out_root()} | PA3_RESULTS={results_root()} | keep={WORK}", flush=True)

    check_disk(args.session)
    errors = {}
    if args.session.startswith("t1"):
        prepare_task1()
    if args.session == "t1-check":
        def check_all(names):
            def step(log, env):
                for v in names:
                    memcheck(v, log, env)
            return step
        errors = run_lanes([("gpu0_check_baseline", [check_all(["full_ft", "full_ft_l2sp"]), eval_step("baseline")]),
                            ("gpu1_check_adapters", [check_all(["lora4", "lora16", "oplora16"])])])
    elif args.session == "t1-fullft":
        errors = run_lanes([("gpu0_full_ft", [train_step("full_ft"), eval_step("full_ft_mid"), eval_step("full_ft")]),
                            ("gpu1_full_ft_l2sp", [train_step("full_ft_l2sp"), eval_step("full_ft_l2sp")])])
    elif args.session == "t1-lora":
        errors = run_lanes([("gpu0_lora4", [train_step("lora4"), eval_step("lora4")]),
                            ("gpu1_lora16", [train_step("lora16"), eval_step("lora16")])])
    elif args.session == "t1-final":
        merge_attached_results()
        attached = link_attached_runs()
        print("attached runs:", sorted(attached) or "none", flush=True)
        errors = run_lanes([("gpu0_oplora16", [train_step("oplora16"), eval_step("oplora16")]),
                            ("gpu1_remaining", [remaining_evals_step(exclude={"oplora16"})])])
        finalize_task1()
    elif args.session == "t2":
        log = LOG_DIR / "t2_features.log"
        run([sys.executable, "-m", "task2.cache_features"], log, lane_env(0 if n_gpus() else None))
        lanes = [("gpu0_vis196", [t2_train("vis196", 196)])]
        if not args.no_vis49:
            lanes.append(("gpu1_vis49", [t2_train("vis49", 49)]))
        errors = run_lanes(lanes)
        alog = LOG_DIR / "t2_analyze.log"
        if "gpu0_vis196" not in errors:
            run([sys.executable, "-m", "task2.analyze", "--run", "vis196"]
                + ([] if args.no_vis49 or "gpu1_vis49" in errors else ["--compare", "vis49"]), alog, lane_env(None))
    print(f"session {args.session} finished in {hours():.2f} h" + (f" with FAILED lanes: {list(errors)}" if errors else ""),
          flush=True)
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
