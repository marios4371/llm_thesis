# MAS-SHT v10.2 — Extensions Quick Reference

This file documents the v10.2 additions on top of the v10.1 pipeline.

## What's new

| Change | Where |
|---|---|
| HuggingFace router, Together AI, local transformers providers | `Mas_solver.py` → `UnifiedLLMClient` |
| Six small-open-model presets | `Mas_solver.py` → `HETEROGENEOUS_PRESETS` |
| `enable_siv` / `enable_sht` ablation flags | `Mas_solver.py` → `QualityAwarePipeline` |
| B1–B4 baseline systems | `baselines.py` |
| Comparative metrics + McNemar + plots | `evaluation_metrics.py` |
| End-to-end Colab harness | `MAS_SHT_Experiments.ipynb` |
| Unit tests | `test_baselines.py`, `test_metrics.py` |

## How to run an experiment from scratch

1. Open `MAS_SHT_Experiments.ipynb` in Google Colab.
2. Add your secrets in **Settings → Secrets**: `GROQ_API_KEY`, `GOOGLE_API_KEY`, `HF_API_KEY`, `TOGETHER_API_KEY`. Missing keys only block presets that use them.
3. Edit cell 1: replace `https://github.com/USERNAME/MAS_LLM_Thesis.git` with your actual repo URL.
4. Edit cell 2 `DATASET_CONFIG` to choose sample size (`{'mode': 'random', 'n': 50, 'seed': 42}` is the safe default for free tiers).
5. Edit cell 3 `CONFIG` to choose which baselines and which MAS variants to run. The default config compares B1, B2, B3, B4 against B5 (NoSIV), B6 (NoSHT), and B7 (full) on `homogeneous_groq`.
6. Run all cells top to bottom.

Outputs land in `/content/drive/MyDrive/MAS_SHT/`:

```
results/        per-system per-run CSVs (one row per problem)
checkpoints/    pickle checkpoints for resume-after-timeout
artifacts/      comparison_table.{csv,md}, mcnemar_results.csv, comparison.png
```

If Colab times out mid-run, just re-run cell 3 — completed problems are skipped.

## How to read the comparison table

`artifacts/comparison_table.csv` has one row per system. Key columns:

- **accuracy** — fraction correct on the apples-to-apples problem intersection.
- **n_paired** — number of problems shared across all systems (use this, not `n_total`).
- **delta_vs_ref** — `system_acc − reference_acc`. Positive ⇒ this system beats the reference.
- **error_reduction_vs_ref** — `(system_acc − ref_acc) / (1 − ref_acc)`. Fraction of the reference's remaining error budget closed by the candidate.
- **avg_llm_calls** — mean LLM calls per problem. Drives the cost axis.
- **accuracy_per_call** — efficiency proxy.
- **siv_trigger_rate** / **siv_skip_rate** / **sht_trigger_rate** — only meaningful for MAS systems.

`artifacts/mcnemar_results.csv`:

- **a/b/c/d** — paired contingency: `a`=both correct, `b`=ref correct only, `c`=other correct only, `d`=both wrong.
- **test_used** — `exact_binomial` when `b+c < 25`, `asymptotic_yates` otherwise, `degenerate_no_discordant` if every problem agreed.
- **p_value**, **significant_at_alpha** — α=0.05 by default.

## Local development

```bash
python -m pytest test_baselines.py test_metrics.py test_siv.py test_integration.py -v
python -m py_compile Mas_solver.py baselines.py evaluation_metrics.py
```

## Known caveats

- 7B HF serverless models are no longer free — use Together (`small_math_homogeneous`, `qwen_math_7b`) or local mode (`tiny_math_homogeneous`).
- Local-HF presets need `transformers` + `accelerate` + `torch`. The Colab notebook installs them; for local dev install on demand.
- TokenBudget is Groq-only. HF/Together/local don't decrement it.
- The v10.1 confidence-gate ordering invariant (baseline-disagreement before SIV-skip) is preserved. The `enable_siv=False` flag only removes SIV's signal — it does not change the gate's ordering.
