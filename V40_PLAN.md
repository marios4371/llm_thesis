# v40: Cross-model Symbolic Surgery — does v38's ADAPT-RS help solvers of other families?

**Files:**
- `cross_model_surgery.py`: what the frozen audit reads (`clean`), which of its FALSE verdicts count (`complete`), the final-answer parser, and the verifier-free arms SC5 / SC5-R / SC5-RS. Pure functions, no model.
- `ollama_client.py`: a standard-library Ollama client. It draws seeded samples and resumes through assistant prefill.
- `pretest_v40.py`: the manifest, the runs (local Ollama, Kaggle HF, stub), the pilot report, PRM scoring and the pre-registered reading.
- `replay_v40.py`: the offline evidence in section 2 (no GPU, no model).
- `test_v40.py`: offline checks, including the frozen sha256 below.
- `pretest_data/v40_cross_model.json`: v38's 200 main rows plus 10 pilot rows from v36, seed 40.
- `MAS_SHT_Kaggle_v40.ipynb` runs the 7–8B models, one per GPU. `MAS_SHT_Kaggle_v40_prm.ipynb` adds the PRM scores that ADAPT and ADAPT-RS need.
- Frozen for v40 (sha256 of the LF text):
  - `cross_model_surgery.py`: sha256 **ad9800c84cc2**;
  - `pretest_data/v40_cross_model.json`: sha256 **9d5d1e14447f**.
- Unchanged: `trace_siv.py` (v35, **483ed23b89b6**) and `trace_surgery.py` (v38, **42376931d94a**). The repair policy is v38's frozen `surgery_policy`.

## 1. Why

v38 measured **ADAPT-RS** on one solver, Qwen2.5-Math-7B-Instruct. ADAPT-RS is v22's verifier-gated sampling (3 samples, 5 when the PRM doubts) plus Symbolic Surgery. On 200 fresh clean GSM-Hard rows it scored 131 vs ADAPT-R 122, W9 L0, p = 0.004. A reader's first question is whether that comes from the method or from one model's LaTeX habits.

v40 runs **the same system** on the same 200 rows with solvers of other families and sizes:
- the same verifier, Qwen2.5-Math-PRM-7B;
- the same gate (min step reward ≥ 0.85 on the best of the first 3);
- the same frozen surgery policy.

The laptop cannot hold the PRM, so every sample, repair and resume is scored afterwards on Kaggle (`--score-prm`). Drawing all 5 samples and reading the gate off the first 3 gives exactly the adaptive procedure; this is how v38 ran too.

## 2. Offline evidence (`python replay_v40.py`; stored samples; canonical grader)

### 2.1 What had to change for other models

The frozen audit reads Qwen-Math's LaTeX well. Two problems appear on chat-style text:
- **Formatting it cannot read.** Markdown bold (`= **984**`), `÷`, `4 x 6`, and money written as `$8` in plain text all yield **no claims at all** (tested 2026-10-22). `clean()` rewrites them. Lines that contain a LaTeX command are left alone.
- **Claims cut out of longer expressions.** The audit's plain-text path takes `2/4/2 = 1` from `2/4/2 = 1/4`, `2.5 + 12 + 2.5 = 30` from `6 + x + 2.5 + 12 + 2.5 = 30`, and `4 + 60/60 = 4` from `4 + 60/60 = 4 + 1`. Each looks false, so it refutes right samples. `complete()` keeps a plain-text FALSE verdict only when no operator continues the claim on either side.

| solver (stored samples) | samples | right | samples with a checked claim, v38 → v40 | refuted | refuted RIGHT |
|---|---|---|---|---|---|
| Qwen2.5-Math-7B, the 200 v40 main rows (v38 run) | 1000 | 566 | 910 → 910 | 108 → 108 | 6 → 6 |
| Qwen2.5-Math-7B, 60 MATH-500 rows (v38 run) | 300 | 254 | 276 → 276 | 15 → 15 | 1 → 1 |
| DeepSeek-Math-7B-RL, P2 dev rows (v31 juror) | 540 | 250 | 457 → 500 | 74 → 38 | **25 → 1** |
| OLMo-2-1124-7B-Instruct, P2 dev rows (v31) | 80 | 17 | 54 → 58 | 10 → 8 | **4 → 1** |

On Qwen-Math nothing changes. On the foreign models the false alarms disappear and coverage rises.

### 2.2 The arms on the same samples (v38's own PRM scores for Qwen-Math)

| solver | rows | SC5 | SC5-R | SC5-RS | SC5-RS vs SC5 | SC5-RS vs SC5-R |
|---|---|---|---|---|---|---|
| Qwen2.5-Math-7B, **the 200 v40 main rows** | 200 | 118 | 120 | **129** | **W12 L1** (p = 0.003) | W10 L1 (p = 0.012) |
| Qwen2.5-Math-7B, MATH-500 | 60 | 52 | 53 | 53 | W1 L0 | W0 L0 |
| DeepSeek-Math-7B-RL, P2 dev | 108 | 59 | 60 | 61 | W2 L0 | W2 L1 |
| OLMo-2-7B, P2 dev | 16 | 4 | 4 | 4 | W0 L0 | W0 L0 |

**With the verifier, Qwen-Math on the 200 main rows** (v40's reading, v38's PRM scores):

| ADAPT | ADAPT-R | **ADAPT-RS** | ADAPT-RS vs ADAPT-R | ADAPT-RS vs ADAPT | ADAPT-RS vs SC5 | calls/row |
|---|---|---|---|---|---|---|
| 116 | 122 | **131** | **W9 L0** | W15 L0 | W18 L5 | 3.46 |

These are exactly v38's numbers: the cleaning and the completeness filter leave Qwen-Math's reading unchanged. For this replay, v40's repaired texts inherit the score v38 gave the same correction. They differ by markdown only, and the run scores its own texts.

- **Qwen-Math, the main rows, without a verifier.** The verifier-free repair already beats the plain vote by 11 rows. The v38 run stored a resume for 104 of the 107 repairable samples.
- **The foreign models.** No resumes were ever drawn for them, so these numbers are a lower bound. They also come from P2 rows, where most errors are readings rather than arithmetic and the repair has little to act on. v40 measures them where it can act: the arithmetic-heavy GSM-Hard rows.

## 3. The method

Per row and model:
- **Samples.** v38's sampler: 5 samples of v22's CoT prompt, T 0.8, top-p 0.95, top-k 50, 1536 new tokens. Each (model, row, sample) has a fixed seed. On Ollama, `repeat_penalty` is 1.0 and `num_ctx` is 2560. Thinking models are asked not to think.
- **Reading each sample.** `clean` → `audit` (frozen trace_siv) → `complete` → `operate` (frozen trace_surgery). The site, the first false claim, must itself be a complete claim.
- **Resumes.** Every sample that v38's frozen policy sends to RESUME, computed over the five, gets one partial call from the corrected prefix. The call uses the same sampling and the rest of the token budget. On Ollama the conversation ends with the open assistant prefix. A model whose template cannot prefill fails a smoke test before its first row, and it is then run without resumes; its RESUME decisions drop the sample.
- **Everything else is recomputed at reading time** from the stored text with the frozen module. The only stored model output besides the samples is the resumes.

**The arms** (canonical grader, ties to the group seen first):
- **SC5:** the plurality of the 5 answers.
- **SC5-R:** the plurality of the unrefuted samples; if every sample is refuted, SC5.
- **SC5-RS:** SC5-R plus the repairs that v38's policy chooses: propagate when the correction reaches the answer, resume when the slip recurs in at least 2 samples, otherwise drop. A repair that is still refuted is dropped. If the pool is empty, SC5. Calls are 5 plus the resumes drawn.

## 4. Pre-registered (FROZEN 2026-10-22, before any main-row sample exists)

- **Rows.** The 200 main rows. A model enters the reading with **at least 100 complete main rows**, and its analysis set is its complete rows. Rows run in manifest order and the stop is by time, never by outcome.
- **Pooling.** The **new** models are pooled. The Qwen-Math reference is reported but not pooled, because it is in-sample for the method.
- **Scores.** A model is read only once its samples, repairs and resumes carry PRM scores (`--score-prm`).
- **PRIMARY.** v38's comparison, **ADAPT-RS vs ADAPT-R**, W/L pooled over the new models.
  - **SUPPORTED:** net ≥ **+5**, W ≥ 2L and two-sided sign p < **0.05**.
  - **REFUTED:** net ≤ 0.
  - **INCONCLUSIVE:** anything else.
- **SECONDARY A.** **CONSISTENT** if every included new model has net ≥ 0 (ADAPT-RS vs ADAPT-R).
- **SECONDARY B.** **ADAPT-RS vs SC5**, self-consistency at 5 calls, pooled: net ≥ **+5** with W ≥ 2L → MET.
- **Model list.** It is frozen after the pilot (section 5). The pilot reads speed, format and audit coverage, **never accuracy**. A pilot model stays in the run only if:
  - (a) its pilot time projects to fit the weekend;
  - (b) at most 10% of its pilot samples have no parsable answer;
  - (c) at least 30% of its pilot samples have a checked claim.

  A model that fails (b) or (c) is reported as out of scope, with its numbers. The claim is therefore about solvers whose written arithmetic the audit can read, and it says so.
- **Reported, not decisive:**
  - per-model accuracies and W/L;
  - audit coverage, refuted samples, refuted right samples, repairs, resumes and calls per row;
  - unparsed and truncated samples, tokens per sample, seconds per row;
  - ADAPT-RS vs ADAPT, B5, and the verifier-free SC5-R / SC5-RS on the same samples.

## 5. Models (candidates; the list is frozen after the pilot)

| where | model | family | size on disk | note |
|---|---|---|---|---|
| laptop (Ollama) | `llama3.2:3b` | Meta | ~2.0 GB | |
| laptop (Ollama) | `phi4-mini` (or `phi3.5`) | Microsoft | ~2.5 GB | |
| laptop (Ollama) | `gemma3:4b` (or `gemma2:2b`) | Google | ~3.3 GB | may not fit whole in 4 GB of VRAM |
| laptop (Ollama) | `qwen3:4b` (or `qwen2.5:3b`) | Alibaba, general purpose | ~2.5 GB | thinking off |
| laptop (Ollama) | `qwen3:8b` (already pulled) | Alibaba | 5.2 GB | does not fit in 4 GB of VRAM; pilot only, to measure |
| Kaggle (HF 4-bit) | `deepseek-ai/deepseek-math-7b-rl` | DeepSeek | | loaded in v31 |
| Kaggle (HF 4-bit) | `ibm-granite/granite-3.1-8b-instruct` | IBM | | loaded in v32 |

The laptop has a GTX 1650 with 4 GB of VRAM and 7.8 GB of RAM. Models of 4B parameters or less fit on the GPU; 7–8B models run mostly on the CPU.

### 5.1 FROZEN 2026-10-26 after the pilot (10 v36 rows per model; Ollama 0.40.8; no accuracy read)

| model | s/row | tok/s | tokens/sample | truncated | unparsed | checked | prefill | hours for 200 rows | decision |
|---|---|---|---|---|---|---|---|---|---|
| `llama3.2:3b` (Meta) | 35 | 45.6 | 320 | 2% | 0% | 60% | yes | 2.0 | **IN** |
| `phi4-mini` (Microsoft) | 70 | 17.8 | 248 | 0% | 0% | 58% | yes | 3.9 | **IN** |
| `gemma3:4b` (Google) | 320 | 7.7 | 491 | 12% | 0% | 90% | yes | 17.8 | **IN, last**: it gets the remaining hours (~12 h ≈ 135 rows, above the 100-row inclusion bar) |
| `qwen3:4b` (Alibaba) | 493 | 14.2 | 1399 | 64% | 0% | 92% | yes | 27.4 | **OUT** by criterion (a): it does not fit the laptop budget; 64% of its samples hit the 1536-token cap even with thinking off |

- **Criteria (b) and (c).** Every candidate passes them: no unparsed answers, and 58–92% of samples carry a checked claim.
- **Run order:** llama3.2:3b → phi4-mini → gemma3:4b. The laptop budget is `--max-hours 18`, decided before any main-row sample.
- **Kaggle:** DeepSeek-Math-7B-RL and Granite-3.1-8B-Instruct, as planned.

## 6. How to run

**Thursday–Friday, once:** pull the candidate models (`ollama pull llama3.2:3b`, …).
- Set the system environment variable `OLLAMA_NUM_PARALLEL=2`, then quit and restart the Ollama app.
- Set Windows to *never sleep when plugged in*.
- Close the browser and other heavy applications during the runs.

**Pilot, Friday evening** (10 rows per model, about 45 minutes, separate files `pretest_v40_pilot_*.json`):

    python pretest_v40.py --backend ollama --pilot --models llama3.2:3b,phi4-mini,gemma3:4b,qwen3:4b,qwen3:8b

It prints s/row, tokens/s, the share of truncated, unparsed and checked samples, and the projected hours for 200 rows. Record the frozen list in section 5, then run.

**The weekend run** (sequential per model; saves after every row; re-running the identical command resumes):

    python pretest_v40.py --backend ollama --models <frozen list> --max-hours 44

**Kaggle, at the same time.** Commit and push first; the notebook clones `main`. `MAS_SHT_Kaggle_v40.ipynb` runs on T4 x2, one model per GPU, for 8.5 h. Download `pretest_v40_*.json`.

**Monday (required for the verdict).** Upload every `pretest_v40_*.json` as one Kaggle Dataset and run `MAS_SHT_Kaggle_v40_prm.ipynb`, about 2 h on T4 x2. Bring back the `pretest_v40_prm_*.json` files and put them next to the result files.

**Reading** (no GPU):

    python pretest_v40.py --summary-only

It reads every `pretest_v40_*.json` in the folder, plus the v38 files for the Qwen-Math reference row.

## 7. What each outcome would mean

- **SUPPORTED and CONSISTENT.** v38's system generalizes: the surgery adds to verifier-gated sampling on families that were never used to design it. This goes into the paper next to v38.
- **SUPPORTED, not consistent.** It helps on average. The models where it hurts say where the audit's reading or the Qwen PRM fails on foreign text. Those models are reported by name.
- **INCONCLUSIVE.** The effect is too small to see on about 200 rows per model. The per-model table and the verifier-free arms still bound it.
- **REFUTED.** The v38 gain is tied to Qwen-Math, plausibly its LaTeX, its carry habits, or a PRM of the same family. This is recorded honestly, and the paper's surgery claim is scoped to that solver.

## 8. Caveats (known before the run)

- **Quantization.** Ollama runs GGUF Q4_K_M and Kaggle runs NF4 4-bit; neither is the published bf16 model. Every comparison is within a model, on the same samples.
- **Rows.** The 200 rows are fresh for every new model. For Qwen-Math they were v38's confirmation rows, which is why it is a reference and not pooled.
- **Where `clean`/`complete` came from.** They were designed on v31's DeepSeek-Math and OLMo samples (P2 rows, not these rows). DeepSeek-Math on Kaggle is therefore partly in-sample for that design. The other families are not.
- **The verifier is Qwen-family.** Qwen2.5-Math-PRM-7B and its 0.85 gate were calibrated on Qwen-Math traces. On other families the gate and the pick may behave differently. That is part of what v40 measures (ADAPT alone is reported per model).
- **Thinking models** run with thinking off, which is not how they are usually deployed.
- **Seeds** make the Ollama runs reproducible on the same machine and Ollama version (recorded in each file).
