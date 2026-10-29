"""Generate the Kaggle notebooks in notebooks/kaggle/ (one per session; see scripts/kaggle_run.py).

  python scripts/make_kaggle_notebooks.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

SESSIONS = {
    "t1_check": ("t1-check", "Task 1 · memory check + baseline evaluation (~1 h)",
                 "Checks that every Task 1 variant fits in T4 memory on its largest batches and picks the memory "
                 "settings, then evaluates the untouched baseline. Run this first: it validates lm-eval and the "
                 "GPU setup in about an hour before you spend many hours on training."),
    "t1_fullft": ("t1-fullft", "Task 1 · Full FT and Full FT + L2-SP (~5–8 h)",
                  "GPU 0 trains `full_ft`, then evaluates `full_ft_mid` and `full_ft`. GPU 1 trains `full_ft_l2sp`, "
                  "then evaluates it. If the run stops with a disk-space message, set `EXTRA = '--lane 0'`, save a "
                  "version, then make a copy with `EXTRA = '--lane 1'`."),
    "t1_lora": ("t1-lora", "Task 1 · LoRA-4 and LoRA-16 (~4–6 h)",
                "GPU 0 trains and evaluates `lora4`; GPU 1 trains and evaluates `lora16`."),
    "t1_final": ("t1-final", "Task 1 · OPLoRA-16 + diagnostics, tables and figures (~4–6 h)",
                 "**Before running: Add Input → Your Work → the saved versions of the `t1_fullft` and `t1_lora` "
                 "notebooks (and `t1_check`).** GPU 0 trains and evaluates `oplora16`; GPU 1 evaluates any "
                 "condition still missing. Then diagnostics, qualitative collection and the report tables/figures."),
    "t2": ("t2", "Task 2 · VLM: features, training (196 and optional 49 tokens), evaluation (~3–5 h)",
           "Downloads the 9.3k selected COCO images, caches SigLIP features, trains the 196-token run on GPU 0 and "
           "the optional 49-token run on GPU 1 at the same time, then evaluates and analyzes both. "
           "Set `EXTRA = '--no_vis49'` to skip the optional run."),
    "t3_live": ("t3-live", "Task 3 · live MATH run: single vs parallel vs refinement (~2–5 h, one T4)",
                "Runs Qwen2.5-3B-Instruct (pinned revision, FP16, batch size 1) on the 50 live questions: one "
                "3,072-token response, three 1,024-token responses with self-consistency, and a critique-and-revision "
                "of the first parallel response. Needs only the repo: `results/task3/math/live_problems.json` must "
                "already be pushed (from `task3.data` on Colab). Every call is saved; if the session stops, run a new "
                "version and it continues from the saved calls. Uses one GPU (T4 x2 or T4 x1 both fine)."),
}

SETTINGS_MD = """## Before you click *Save Version → Save & Run All (Commit)*
1. **Settings → Accelerator: GPU T4 x2** (not P100: recent PyTorch builds no longer support it).
2. **Settings → Internet: On** (needs a phone-verified Kaggle account).
3. **Add-ons → Secrets → `GITHUB_TOKEN`** attached to this notebook: a GitHub fine-grained token with
   *Contents: Read and write* on `adilawan1/ATML-PA3`. Without it the run still works; download
   `results/` from the Output tab instead.
4. Choose **Save & Run All (Commit)**, not Quick Save: the run then continues with the browser closed (up to 12 h)."""


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.strip("\n")}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n")}


def notebook(session, title, what):
    cells = [
        md(f"# ATML PA3 on Kaggle — {title}\n\n{what}\n\n{SETTINGS_MD}"),
        code(f"""
import os, shutil, subprocess
from pathlib import Path

SESSION = '{session}'
EXTRA = ''   # e.g. '--lane 0' (one GPU lane only) or '--no_vis49' (Task 2)
REPO_URL = 'https://github.com/adilawan1/ATML-PA3.git'

scratch = Path('/kaggle/temp' if os.path.isdir('/kaggle/temp') else '/tmp') / 'pa3'   # big disk, not saved
os.environ.update(
    PA3_SCRATCH=str(scratch), PA3_OUT=str(scratch / 'outputs'), PA3_COCO=str(scratch / 'coco'),
    HF_HOME=str(scratch / 'hf_cache'),
    PA3_RESULTS='/kaggle/working/results',   # small results: saved with the version and pushed to GitHub
    PA3_KEEP='/kaggle/working/outputs',      # weights later sessions need (attach this notebook's output)
    PA3_LOGS='/kaggle/working/logs',
    PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', TOKENIZERS_PARALLELISM='false')
REPO = scratch / 'ATML-PA3'
scratch.mkdir(parents=True, exist_ok=True)
if not REPO.exists():
    subprocess.run(['git', 'clone', '-q', REPO_URL, str(REPO)], check=True)
subprocess.run(['git', '-C', str(REPO), 'pull', '-q'], check=True)
print(subprocess.run(['git', '-C', str(REPO), 'log', '-1', '--format=%h %s'], capture_output=True, text=True).stdout)
!nvidia-smi --query-gpu=index,name,memory.total --format=csv
"""),
        code("""
# Kaggle's preinstalled CUDA PyTorch satisfies torch>=2.4, so it is kept; the rest is installed/pinned.
!pip install -q -r {REPO}/requirements.txt 2>&1 | grep -v -i "dependency resolver\\|incompatible" | tail -5
import torch, transformers
print('torch', torch.__version__, '| transformers', transformers.__version__, '| GPUs', torch.cuda.device_count())
assert torch.cuda.is_available(), 'No GPU: set Accelerator to GPU T4 x2'

# start from the committed results (never overwrite files already in /kaggle/working/results)
res = Path(os.environ['PA3_RESULTS'])
for f in (REPO / 'results').rglob('*'):
    t = res / f.relative_to(REPO / 'results')
    if f.is_file() and not t.exists():
        t.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, t)
"""),
        md("## Run the session\nProgress is printed every 10 minutes; full logs are in `/kaggle/working/logs/`."),
        code("""
!cd {REPO} && python scripts/kaggle_run.py {SESSION} {EXTRA}
rc = _exit_code
print('SESSION OK' if rc == 0 else f'SESSION FAILED (exit {rc}): check /kaggle/working/logs; results so far are still saved')
"""),
        md("## Save results to GitHub\nRuns even if the session failed, so partial results are kept."),
        code("""
def push_results():
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret('GITHUB_TOKEN')
    except Exception:
        token = None
    if not token:
        print('No GITHUB_TOKEN secret: download results/ from the Output tab and commit it yourself.')
        return
    g = ['git', '-C', str(REPO), '-c', 'user.name=adilawan1', '-c', 'user.email=adilawan1@users.noreply.github.com']
    shutil.copytree(os.environ['PA3_RESULTS'], REPO / 'results', dirs_exist_ok=True)
    subprocess.run(g + ['add', 'results'], check=True)
    if subprocess.run(g + ['diff', '--cached', '--quiet']).returncode == 0:
        print('nothing new to commit'); return
    subprocess.run(g + ['commit', '-qm', f'Kaggle {SESSION} results'], check=True)
    url = f'https://adilawan1:{token}@github.com/adilawan1/ATML-PA3.git'
    for attempt in range(3):   # another session may have pushed meanwhile
        r = subprocess.run(g + ['pull', '-q', '--rebase', url, 'main'], capture_output=True, text=True)
        if r.returncode != 0:   # never print the token
            print('pull failed:', r.stderr.replace(token, '***')[-300:])
            subprocess.run(g + ['rebase', '--abort'], capture_output=True)
        r = subprocess.run(g + ['push', '-q', url, 'HEAD:main'], capture_output=True, text=True)
        if r.returncode == 0:
            print('pushed results to GitHub'); return
        print('push failed:', r.stderr.replace(token, '***')[-300:])
    print('Could not push; download results/ from the Output tab.')

push_results()
"""),
        code("""
# last lines of each log
for p in sorted(Path(os.environ['PA3_LOGS']).glob('*.log')):
    lines = p.read_text(errors='ignore').replace('\\r', '\\n').splitlines()
    print(f'===== {p.name}'); print('\\n'.join(lines[-15:]))
!du -sh /kaggle/working/* 2>/dev/null
"""),
    ]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python"},
                                         "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True,
                                                    "isGpuEnabled": True}},
            "nbformat": 4, "nbformat_minor": 4}


def main():
    out = REPO / "notebooks/kaggle"
    out.mkdir(parents=True, exist_ok=True)
    for name, (session, title, what) in SESSIONS.items():
        (out / f"pa3_{name}.ipynb").write_text(json.dumps(notebook(session, title, what), indent=1) + "\n")
    print("wrote", sorted(p.name for p in out.glob("*.ipynb")))


if __name__ == "__main__":
    main()
