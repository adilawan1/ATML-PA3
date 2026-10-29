# PA3 Progress Log

Status legend: ✅ done (real result produced) · 🧪 code written + smoke-tested (CPU, tiny model / synthetic data) · ⏳ needs a GPU run on Colab · ❓ needs input or verification from you

_Last updated: 2026-10-25 (autonomous session)_

## Where things stand

All three tasks have complete pipelines that run end to end in CPU smoke tests: data → training → evaluation → tables/figures. No GPU work has been done; local runs were only smoke tests. The real data selections for Task 1 and Task 2 are deterministic and were produced locally; Task 3 is waiting on the supplied rollout data.

**Next actions for you**
1. ✅ Public repo created and pushed: https://github.com/adilawan1/ATML-PA3 (`REPO` set in the notebooks, the report abstract link is set).
2. ❓ Get the `ATML-PA3-Data` rollouts (not on the HF Hub under that name). Then run `python -m task3.data --dataset math --inspect`, check the column names against `configs/task3.yaml → columns`, and set `live.revision` and `live.separator` from the supplied README.MD.
3. On Colab, run `notebooks/task1_finetuning.ipynb` first; it's the longest job. Task 3 offline analysis can run on a CPU runtime (or locally) in parallel.

## Disconnect safety (added 2026-10-26)
- Checkpoint `state.pt` is written last and atomically (temp file + rename); only checkpoints with a complete `state.pt` are resumed from.
- On resume, `train_log.jsonl` (and the Task 2 `dev_log.jsonl`) are trimmed to the checkpoint's update, so updates are never logged twice.
- Notebooks set `PA3_RESULTS` to Drive (`MyDrive/ATML-PA3/results`), seeded from the committed `results/`; the save cell rsyncs Drive → repo before committing. A disconnect therefore loses at most the training since the last checkpoint and the condition being evaluated.

## Task 2 completed on Colab A100 (2026-10-29)
- vis196 and the optional vis49 run trained (4,500 updates each, ~45 min) and were evaluated on the A100.
- FP16 overflow at the start: updates 1–5 (vis196) and 1–3 (vis49) were skipped by the GradScaler, and the interface check passed at update 6 / 4. vis49 had one more skip at update 4,411. No non-finite losses.
- Connector scale = 0.0549 (text-embedding norm 0.870 / initial connector-output norm 15.87).
- Feature caching took 37 s for 9,258 images; the images were already on disk, so this run did not time the download.
- Logs re-pushed from Drive overwrote the cleaned Task 1 logs and figures (Drive still had old copies). Restored. To fix the Drive copy, run `python -m task1.analyze` once in Colab after `git pull`.

## Task 1 completed on Colab A100 (2026-10-29)
- All 5 trainings and all 7 evaluations ran on Colab A100-SXM4-40GB, **except the baseline evaluation, which ran on an L4** (scores unaffected in principle; GSM8K time 29.5 min vs 8–12 min on the A100). Re-run it on the A100 if evaluation times are compared.
- LoRA-4 and LoRA-16 were each restarted from scratch before their first checkpoint (248 and 21 extra log rows). The old trainer kept those rows; committed logs now hold only the final attempt (`results/task1/runs/log_cleaning.json`), and both trainers now clear the log on a fresh start. Benchmark scores and summaries were unaffected.
- GSM8K strict-match is 0 for every condition: the `gsm8k_cot` strict regex needs "The answer is N.", while fine-tuned models write "The answer is: N" (MetaMathQA style) and the baseline rarely uses the phrase. Report flexible-extract only, and say why.
- BoolQ changes follow a yes/no bias shift: predicted-"yes" rate 57.7% (baseline) → 27.1% (Full FT), 42.6% (L2-SP), 54–64% (LoRA/OPLoRA); gold is 62.2%.
- Remaining: 42-row manual annotation (`results/task1/qualitative/annotation_sheet.csv`).

## Kaggle runs (added 2026-10-28)
Colab Pro L4 sessions kept disconnecting, so Task 1 and Task 2 move to Kaggle (2 x T4, background "Save & Run All" versions, 12 h limit, ~30 GPU h/week). See `notebooks/kaggle/` and `scripts/kaggle_run.py`.
- All Task 1 training and evaluation runs on the same GPU type (T4), so timings stay comparable. Two runs share a session, one per GPU; note this in the resources discussion.
- New memory options in `task1/train.py` (defaults unchanged): `--loss_chunk` (LM head + CE in checkpointed blocks; verified identical loss/gradients on a tiny model up to float rounding) and `--theta0_device cpu` (L2-SP reference in pinned host memory; penalty time then includes the host-to-GPU copy). `task1.memcheck` picks the cheapest setting that fits per variant; the choice is saved in `results/task1/kaggle/memory_<variant>.json`. Report whichever was used.
- Only what later steps need is kept in `/kaggle/working/outputs` (weights at 625/1,250, adapters, bases, logs). Optimizer state stays on scratch disk, so a Kaggle run cannot be resumed in a later session; each session fits in 12 h instead.

## GPU run plan (Colab Pro; use one GPU type for all timed runs, ideally L4)
Times are rough estimates for an L4, not measurements.
| Order | Job | Notebook cell | Est. time | Notes |
|---|---|---|---|---|
| 1 | Task 1 baseline export + eval + qualitative | task1 §3 (first condition) | 30–60 min | Validates the lm-eval setup early; check `results/task1/eval/baseline/scores.json` looks sane (Qwen3-0.6B non-thinking GSM8K should be well above 0) |
| 2 | Task 1 `full_ft` | task1 §2 | 1–1.5 h | Watch for skipped steps / NaN in the first 20 updates |
| 3 | Task 1 `full_ft_l2sp`, `lora4`, `lora16`, `oplora16` | task1 §2 | ~1 h each | Re-run a cell to resume after a disconnect |
| 4 | Task 1 remaining 6 conditions | task1 §3 | 3–5 h | Set `eval.backend: vllm` to speed up GSM8K if vLLM installs cleanly |
| 5 | Task 1 diagnostics + analysis | task1 §4 | 10 min | |
| 6 | Task 2 caching + training + eval | task2 | 1.5–2.5 h | Image download ≈ 9.3k files |
| 7 | Task 3 live run | task3 | 1–2 h | Batch size 1, sampling |
| — | Task 3 offline | task3 | ~15 min CPU | Can run locally |

## Environment
- Local: Windows, no GPU. `.venv` (CPU torch 2.14, transformers 5.20.2, lm_eval 0.4.13, math-verify 0.9.0) is for smoke tests only.
- Colab: `PA3_OUT` → Drive (checkpoints, caches); `results/` → committed small results.
- Smoke tests: `PA3_OUT=outputs/smoke PA3_RESULTS=outputs/smoke/results` with `configs/*_smoke.yaml`, a random 28-layer tiny Qwen3 (`outputs/tiny/qwen3-tiny`) and synthetic data from `scripts/make_fake_t{2,3}_data.py`.

---

## Task 1: LLM fine-tuning
| Item | Status | Notes |
|---|---|---|
| Data selection `task1/prepare_data.py` | ✅ | 40,000 kept from the first 40,002 MetaMathQA rows. Excluded: 2 too long, 0 empty, 0 GSM8K-test overlaps. 9.91M tokens (7.10M supervised), mean length 248, max 1,941. Token hash `5db9ba4d…`. → `results/task1/data_selection.json` |
| Probe set `task1/sample_probes.py` | ✅ | 32 GSM8K test + 32 HellaSwag val (seed 6304), fixed before any results. Prompt formats match lm-eval's logged prompts. → `results/task1/probe_set.json` |
| Trainer (5 variants) `task1/train.py` | 🧪 ⏳ | Loss normalized by tokens over the effective batch; LM head applied only at supervised positions; FP16 autocast + GradScaler; unscale → L2-SP grad → clip; skipped updates logged; resume verified bit-exact. |
| LoRA / OPLoRA `common/lora.py` | 🧪 | Deterministic per-module init, so LoRA-16 and OPLoRA-16 share initial factors. Projection done in FP32 outside autocast; constraint residual ~3e-7 on CPU smoke tests, ≤1.9e-4 relative on the real A100 run (ρ_k < 1e-7). |
| Export + merge check `task1/export_models.py` | 🧪 ⏳ | Merged vs unmerged logits plus OPLoRA constraint check → `results/task1/export/`. |
| lm-eval runner `task1/evaluate.py` | 🧪 ⏳ | Forces `enable_thinking=False` and asserts `</think>` in the rendered prompt. gsm8k_cot 0-shot, `max_gen_toks=1024`, stock stops `["Q:","</s>","<|im_end|>"]` and filters. Per-example rows include extracted answer, generated tokens and truncation flag. |
| Diagnostics ρ_k, η, R(z) `task1/diagnostics.py` | 🧪 ⏳ | FP32; activations cached from the starting model. Also reports input-subspace ρ and the random baseline k/d = 0.125. |
| Qualitative `task1/qualitative.py` | 🧪 ⏳ | 168 greedy responses; auto flags (JSON validity, template fields, truncation, repetition, math markers); 42-row annotation CSV. |
| Analysis `task1/analyze.py` | 🧪 | Benchmark table (md/csv/tex), resources, regularization, loss/LR/λΩ plots, diagnostics figure; copies run logs into `results/`. |
| GPU runs: 5 trainings, 7 evaluations | ⏳ | — |
| Manual annotation of the 42 responses | ⏳ | Done by hand after the GPU runs. |

**Task 1 decisions (state in the report)**
- Prompt = non-thinking generation prefix (`…<|im_start|>assistant\n<think>\n\n</think>\n\n`); target = response + `<|im_end|>`. Verified identical to the chat template's rendering of a full conversation.
- Cosine schedule decays to 0 (HF `get_cosine_schedule_with_warmup`), so update 1 runs at LR 0 (HF convention).
- L2-SP: gradient λ(θ−θ0) is added after unscaling and before clipping, so clipping applies to the full objective. θ0 is an FP32 GPU copy, logged as reference-storage overhead. Full FT keeps its θ0 copy on the CPU so its memory numbers aren't distorted.
- Only the newest recovery checkpoint keeps optimizer state (Drive space). Every checkpoint's size is logged before pruning. Weights at 625/1250 are kept.
- LoRA: A uses Kaiming-uniform init (a=√5, as PEFT), B = 0. CPU smoke tests use an FP32 backbone only because CPU FP16 is impractically slow.
- GSM8K uses a fixed generation batch of 64 (`auto` can OOM at 1,024 tokens); likelihood tasks use `auto`.

## Task 2: VLM and visual grounding
| Item | Status | Notes |
|---|---|---|
| Official VQA normalization port `task2/vqa_norm.py` | ✅ | Exact tables from GT-Vision-Lab `vqaEval.py`; soft accuracy checked by hand. |
| Data selection `task2/prepare_data.py` | ✅ | Real selection done locally (4 min). 8,730 train2014 images (6,000 caption + VQA-train); 528 val2014 images across 5 mutually disjoint dev/test sets (assertions pass). VQA candidates after filtering — train: 44,033 yes/no, 10,561 number, 17,375 other; val: 20,963 / 4,923 / 8,039. Pools: 20k presence (19,996 unique), 20k counting (16,113 unique). → `results/task2/data/` |
| Feature caching `task2/cache_features.py` | 🧪 ⏳ | Tested on 6 real COCO images with the real SigLIP: `last_hidden_state` == post-LN(final encoder output) (max diff 0.0) and ≠ pre-LN, so LN is applied exactly once. Shape 196×768, FP16 memmap. Preprocessing: resize 224 bicubic, rescale 1/255, normalize (0.5, 0.5, 0.5). Timed separately. Full run ≈ 9.3k images ≈ 2.8 GB. |
| Model `task2/vlm.py` | 🧪 | Connector (768→1024→1024, GELU) plus a fixed calibrated scale. `<|image_pad|>` embeddings replaced in place. No-image input drops the whole vision block. Greedy KV-cache decoding (batched output == single output, verified). |
| Trainer `task2/train.py` | 🧪 ⏳ | 3 stages, seeded mixture sampler (each draw = one pair or two captions), per-stage warmup + cosine to 10%, checkpoint every 250 updates, stage weights, dev eval at 0/500/…/2250/3750/4500. Resume verified bit-exact across a stage boundary. Interface checks per stage: backbone frozen, connector and LoRA grads non-zero. |
| Final evaluation `task2/evaluate.py` | 🧪 ⏳ | 4 conditions; EM / VQA soft accuracy by type; pair accuracy, identical-answer, empty-answer, NLL, ms/answer, partner-follow rate; correct-image advantages; 4,000-resample pair bootstrap; fixed seeded derangement for mismatched images (saved). |
| Analysis `task2/analyze.py` | 🧪 | Curves with stage boundaries, final/type/advantage tables, resources, qualitative figure, optional 49-token comparison. |
| Optional 49-token run | ⏳ (optional) | `--n_vis 49` (2×2 average pooling before the connector). |

**Task 2 decisions (state in the report)**
- Prompts: caption = "Write a short caption for this image."; questions + "\nAnswer the question using a single word or phrase." (LLaVA-1.5 short-answer prompt) Identical across image conditions.
- Grounding split: dev 12 presence + 12 counting pairs; test 32 + 32.
- Annotation-noise mitigation: presence positives need ≥1 non-crowd instance ≥ 0.5% of image area. Counting images need every instance non-crowd and ≥ 0.5% of area, with count 1–4. Negatives have no annotation of the category.
- Training pools: 20k presence + 20k counting pairs pre-generated over the cached train images (caption images ∪ VQA-train images). Category is drawn uniformly over eligible categories.
- VQAv2: both questions of a pair must share an answer type. Canonical target = most frequent original string within the consensus group. Normalization is applied to all answers (the official script skips it when all 10 answers are identical).
- A new AdamW is created at each stage (the parameter groups change); the GradScaler continues across stages. Weight decay 0.01 on 2-D weights only.
- Connector scale rule: mean text-embedding norm ÷ mean connector-output norm at init, computed on training data → `results/task2/vis196/calibration.json`.
- Task 2 warmup starts at LR peak/w (not 0).

## Task 3: Test-time compute
| Item | Status | Notes |
|---|---|---|
| Loader / joins / alignment checks / grouping / grading `task3/data.py`, `task3/answers.py` | 🧪 ❓ | Column names auto-resolve from candidate lists in the config. Symmetric math-verify grouping without gold (cached; SIGALRM timeouts on Linux). MathVista official normalization with MC → letters. Single NO_ANSWER group. Splits = sorted IDs → `default_rng(6304).permutation`. 10 seeded orders per question. |
| Selection rules `task3/selection.py` | 🧪 | FirstAns, SC, max L/K/V/VisConf/tokenwise, RWC with average ranks, shortest-rollout control. VisConf recalibrated per prefix (linear percentiles). Ties go to the earliest supporting rollout. |
| Offline evaluation `task3/offline.py` | 🧪 ⏳ | Accuracy, oracle coverage, distinct groups, token cost, sampling-vs-selection failure decomposition, paired question bootstrap (1,000). About 2–3 min per dataset. |
| Signal analysis `task3/signals.py` | 🧪 ⏳ | Quartile gaps (all and mixed-correctness questions), Spearman correlations, score–length correlation, c–v correlation, numerical check of Cov(c, v), fixed illustrative trajectories. |
| Live run `task3/live.py` | 🧪 ⏳ ❓ | Single 3,072 / parallel 3×1,024 + SC / refinement (shared first response + 2,048). Per-call seeds, tokens, truncation, time, peak memory; resumable; transitions and bootstrap. Needs the problem text column, the model revision and the separator from the README. |
| Adaptive stopping `task3/adaptive.py` | 🧪 ⏳ | Agreement and quality-aware controllers, dev grid with fallback-32, selection rule exactly as specified, held-out results, stop distributions, incorrect early stops recoverable by fixed-32, behaviour by pool correctness. |
| Analysis `task3/analyze.py` | 🧪 | Scaling and cost curves, method table with CIs, failure plot/table, adaptive vs fixed-N, live table, selection-disagreement examples. |

**Task 3 decisions (state in the report)**
- Generated-token cost uses the supplied retained token count column if present, otherwise the length of the token measurements.
- Main results are on the 800 held-out questions; dev (200) is used only for threshold selection.
- Missing attention denominators (0) are excluded from Vᵢ via nanmean.
- Refinement returns the revised answer even when it is missing (counted as NO_ANSWER); no fallback to the initial answer.

## Deliverables checklist
- [ ] Public GitHub repo with code, configs, IDs, environment, results, README (README written ✅)
- [ ] All GPU runs (Task 1 ×5 train + 7 eval; Task 2 main run; Task 3 live)
- [ ] 42-response manual annotation (Task 1)
- [ ] 8-page NeurIPS report (PA0 template), repo link at the end of the abstract
- [ ] LMS submission by 06 Nov 2026
