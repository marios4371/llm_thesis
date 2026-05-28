# MAS-SHT v10.4 — Extensions Quick Reference

This file documents the cumulative extensions on top of the v9.1 paper run.

## What's new in v10.4 (publication-readiness fixes)

| Change | Where | Why |
|---|---|---|
| Removed duplicated `__main__` block (89 lines) | `Mas_solver.py` | Was running the same CLI twice |
| `evaluation_mode=True` forces `use_cache=False` | `Mas_solver.py` → `QualityAwarePipeline` | Cache hits break apples-to-apples |
| Silent distractors (default) + legacy `distractor_labeled` mode | `Mas_solver.py` → `EnhancedProblemManager._maybe_harden` | Old distractors self-announced as "ignore this" — trivial signal |
| Baseline prompt: system+user split, `max_tokens` 500→1024 | `Mas_solver.py` → `solve()`, `baselines.baseline_only` | 500 tokens truncated CoT mid-stream on hard problems |
| `dataset_seed` constructor argument | `Mas_solver.py` → `QualityAwarePipeline` | Reproducibility + multi-seed harness |
| **PAL** (Gao et al., ICML 2023) baseline | `baselines.py` → `pal()` | Strong code-execution baseline — mandatory comparator |
| **PoT** (Chen et al., TMLR 2023) baseline | `baselines.py` → `pot()` | NL + code baseline — second mandatory comparator |
| Wilson 95% CI for accuracy | `evaluation_metrics.py` → `wilson_ci` | Headline CI columns `accuracy_ci_lo/hi` |
| Bootstrap 95% CI for paired delta | `evaluation_metrics.py` → `paired_delta_bootstrap_ci` | Paired CI on Δ-accuracy vs reference |
| 3 strong-model presets (Llama-70B, DeepSeek-Math-7B, Qwen-Math-72B) | `Mas_solver.py` → `HETEROGENEOUS_PRESETS` | Cross-architecture comparison required |
| UTF-8 stdout in test files | `test_siv.py`, `test_integration.py` | Windows console encoding fix |

## What was already in v10.2

| Change | Where |
|---|---|
| HuggingFace router, Together AI, local transformers providers | `Mas_solver.py` → `UnifiedLLMClient` |
| Six small-open-model presets | `Mas_solver.py` → `HETEROGENEOUS_PRESETS` |
| `enable_siv` / `enable_sht` ablation flags | `Mas_solver.py` → `QualityAwarePipeline` |
| B1–B4 baseline systems | `baselines.py` |
| Comparative metrics + McNemar + plots | `evaluation_metrics.py` |
| End-to-end Colab harness | `MAS_SHT_Experiments.ipynb` |
| Unit tests | `test_baselines.py`, `test_metrics.py`, `test_siv.py`, `test_integration.py` |

## How to run a publication-grade experiment

1. Open `MAS_SHT_Experiments.ipynb` in Google Colab (or `MAS_SHT_Kaggle.ipynb` on Kaggle).
2. Add your secrets in **Settings → Secrets**: `GROQ_API_KEY`, `GOOGLE_API_KEY`, `HF_API_KEY`, `TOGETHER_API_KEY`. Missing keys only block presets that use them.
3. Edit cell 1: point the repo URL at your fork.
4. Edit cell 2 `DATASET_CONFIG`. For a publishable run aim for `{'mode': 'random', 'n': 300, 'seed': 42}` per benchmark; re-run with seeds `[42, 43, 44]` for multi-seed error bars.
5. Edit cell 3 `CONFIG` to enable **all** required comparators:
   - `b1_direct`, `b2_cot`, `b3_sc5`, `b4_baseline_only`, **`b_pal`**, **`b_pot`** (NEW)
   - MAS variants: full, no-SIV, no-SHT (ablation)
6. Pass `evaluation_mode=True` to `QualityAwarePipeline` so the cache is forced off.
7. Run all cells top to bottom. Outputs land in `${RESULTS_DIR}/`:

```
results/        per-system per-run CSVs (one row per problem)
checkpoints/    pickle checkpoints for resume-after-timeout
artifacts/      comparison_table.{csv,md}, mcnemar_results.csv, comparison.png
```

If Colab/Kaggle times out, just re-run the runner cell — completed problems are skipped.

## How to read the comparison table

`artifacts/comparison_table.csv` has one row per system. Key columns:

- **accuracy** — fraction correct on the apples-to-apples problem intersection.
- **accuracy_ci_lo / accuracy_ci_hi** — Wilson 95% CI on accuracy. **Quote these in any reported claim.**
- **n_paired** — number of problems shared across all systems (use this, not `n_total`).
- **delta_vs_ref** — `system_acc − reference_acc`. Positive ⇒ this system beats the reference.
- **delta_ci_lo / delta_ci_hi** — Bootstrap 95% CI on the **paired** delta. If `0` is inside the interval, the difference is not significant.
- **error_reduction_vs_ref** — `(system_acc − ref_acc) / (1 − ref_acc)`.
- **avg_llm_calls / avg_tokens / avg_time_s** — cost axes.
- **accuracy_per_call** — efficiency proxy.
- **siv_trigger_rate / siv_skip_rate / sht_trigger_rate** — only meaningful for MAS systems.

`artifacts/mcnemar_results.csv`:

- **a/b/c/d** — paired contingency: `a`=both correct, `b`=ref correct only, `c`=other correct only, `d`=both wrong.
- **test_used** — `exact_binomial` when `b+c < 25`, `asymptotic_yates` otherwise, `degenerate_no_discordant` if every problem agreed.
- **p_value**, **significant_at_alpha** — α=0.05 by default.

## Local development

```bash
# Run all tests (47 total: 16 baselines + 19 metrics + 8 SIV + 4 integration)
python test_baselines.py
python test_metrics.py
python test_siv.py
python test_integration.py

# Syntax check
python -m py_compile Mas_solver.py baselines.py evaluation_metrics.py siv_module.py
```

## Reproducibility checklist for a publishable run

- [ ] `evaluation_mode=True` passed to `QualityAwarePipeline` (cache disabled)
- [ ] `dataset_seed=<int>` set explicitly, NOT `None`
- [ ] `hardener="distractor"` uses the new silent distractors (NOT `"distractor_labeled"`)
- [ ] `b_pal` AND `b_pot` are in `CONFIG['baselines_to_run']`
- [ ] At least one cross-family preset has been run (e.g., `homogeneous_llama70b_groq` in addition to `homogeneous_groq`)
- [ ] Each system has been run on at least 200 problems per benchmark
- [ ] At least 3 seeds have been swept and per-seed CSVs saved
- [ ] McNemar p-values and bootstrap CIs both appear in the writeup

## Known caveats

- 7B HF serverless models are no longer free — use Together (`small_math_homogeneous`, `qwen_math_7b`) or local mode (`tiny_math_homogeneous`).
- Local-HF presets need `transformers` + `accelerate` + `torch`. The Colab notebook installs them; for local dev install on demand.
- TokenBudget is Groq-only. HF/Together/local don't decrement it.
- The v10.1 confidence-gate ordering invariant (baseline-disagreement before SIV-skip) is preserved. The `enable_siv=False` flag only removes SIV's signal — it does not change the gate's ordering.
- The legacy `hardener="distractor_labeled"` mode is kept ONLY for reproducing v10.3 paper numbers. Do not use for new runs.
