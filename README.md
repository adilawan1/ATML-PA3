# ATML PA3: Fine-Tuning, Vision-Language Grounding, and Test-Time Compute

Code for EE-5102/CS-6304 Programming Assignment 3. It has three tasks:

1. **Task 1**: Qwen3-0.6B fine-tuned on MetaMathQA with Full FT, Full FT + L2-SP, LoRA-4, LoRA-16 and OPLoRA-16. Each run is evaluated for math gains and for retention (GSM8K, HellaSwag, BoolQ, ARC-Easy), with weight-space, activation and qualitative diagnostics.
2. **Task 2**: a SigLIP→MLP→Qwen3 VLM trained in three stages (caption alignment, visual instruction tuning, natural VQA). Grounding is tested with complementary pairs and image controls.
3. **Task 3**: test-time compute on the supplied Qwen2.5 / Qwen2.5-VL rollouts. Covers repeated sampling, self-certainty, VisConf, RWC and adaptive stopping, plus a live MATH run comparing parallel sampling, a single longer response and refinement.

Detailed status, decisions and deviations: [progress.md](progress.md).

## Layout

```
common/      utils.py (IO, seeding, RNG state, timing), lora.py (LoRA + OPLoRA, own implementation)
configs/     task1.yaml, task2.yaml, task3.yaml (all hyperparameters; *_smoke.yaml = CPU test configs only)
task1/       prepare_data, train, export_models, evaluate (lm-eval), sample_probes, diagnostics, qualitative, analyze
task2/       prepare_data, cache_features, vlm, data, train, evaluate, vqa_norm, analyze
task3/       data, answers, selection, offline, signals, adaptive, live, analyze
notebooks/   Colab drivers, one per task
scripts/     synthetic-data generators for smoke tests
results/     committed machine-readable results: split/example IDs, logs, scores, tables, figures
data/task1/  first_plot_questions.yaml (qualitative prompts)
```

Large artifacts (checkpoints, tokenized data, feature caches, rollout pools) go under `$PA3_OUT`, which defaults to `./outputs`. On Colab, point it at Drive. Raw datasets and checkpoints are not committed.

## Setup

```bash
pip install -r requirements.txt   # on Colab keep the preinstalled CUDA torch
export PA3_OUT=/content/drive/MyDrive/ATML-PA3/outputs
```

The notebooks in `notebooks/` run each task end to end on Colab. Every training script resumes from its newest checkpoint when re-run.

## Reproducing each task

All randomness uses seed **6304**. Every reported number is produced by the commands below and saved under `results/`.

### Task 1
```bash
python -m task1.prepare_data                 # 40k eligible MetaMathQA examples -> results/task1/data_selection.json
python -m task1.sample_probes                # fixed 32+32 activation-probe set -> results/task1/probe_set.json
python -m task1.train --variant {full_ft,full_ft_l2sp,lora4,lora16,oplora16}
for c in baseline full_ft_mid full_ft full_ft_l2sp lora4 lora16 oplora16; do
  python -m task1.export_models --condition $c --out_dir /content/merged   # merge + merged/unmerged logit check
  python -m task1.evaluate      --condition $c --model_dir /content/merged/$c
  python -m task1.qualitative   --condition $c --model_dir /content/merged/$c
done
python -m task1.diagnostics && python -m task1.qualitative --collect && python -m task1.analyze
```

### Task 2
```bash
python -m task2.prepare_data      # COCO captions / grounding pairs / VQAv2 complementary pairs (IDs saved)
python -m task2.cache_features    # downloads only the selected images; SigLIP features cached in FP16
python -m task2.train             # 4,500 updates over 3 stages (optional: --n_vis 49)
python -m task2.evaluate && python -m task2.analyze
```

### Task 3
```bash
export PA3_T3_DATA=/path/to/ATML-PA3-Data
python -m task3.data --dataset math --inspect     # check column names against configs/task3.yaml
python -m task3.data --dataset {math,mathvista}   # groups, correctness, splits, orders
python -m task3.offline  --dataset {math,mathvista}
python -m task3.signals  --dataset {math,mathvista}
python -m task3.adaptive --dataset {math,mathvista}
python -m task3.live && python -m task3.live --summarize   # GPU
python -m task3.analyze
```

### Smoke tests (CPU, no real models)
Every pipeline was checked end to end with a tiny random Qwen3 and synthetic data. Use `configs/*_smoke.yaml` with `PA3_OUT=outputs/smoke PA3_RESULTS=outputs/smoke/results`, `scripts/make_fake_t2_data.py` and `scripts/make_fake_t3_data.py`.

## Attribution of reused code

- **VQA answer normalization** (`task2/vqa_norm.py`): ported from the official VQA evaluation script ([GT-Vision-Lab/VQA](https://github.com/GT-Vision-Lab/VQA), `vqaEval.py`). The contraction, number and article tables and the punctuation rules are copied unchanged.
- **MathVista answer normalization** (`task3/answers.py::normalize_mathvista`): ported from MathVista's `normalize_extracted_answer` ([lupantech/MathVista](https://github.com/lupantech/MathVista)).
- **HellaSwag preprocessing** (`task1/sample_probes.py::hellaswag_preprocess`): copied from lm-evaluation-harness `tasks/hellaswag/utils.py`.
- **Qualitative prompts** `data/task1/first_plot_questions.yaml`: from [emergent-misalignment/emergent-misalignment](https://github.com/emergent-misalignment/emergent-misalignment) (`evaluation/first_plot_questions.yaml`).
- **LoRA initialization** follows PEFT's default (Kaiming-uniform A, zero B). LoRA and OPLoRA themselves are implemented in `common/lora.py`; OPLoRA follows Xiong & Xie, Sections 3.2–3.3.
- **Libraries**: PyTorch, Hugging Face Transformers / Datasets, lm-evaluation-harness (benchmark scoring), math-verify (MATH equivalence), NumPy/SciPy/pandas/matplotlib.
- Code was written with help from an LLM coding assistant (Claude). All of it was reviewed and tested.
