# v22: the Verifier agent (status as of 2026-09-27)

This is the current state of the thesis experiments. The main MAS-SHT solver
(`Mas_solver.py`, `SOLVER_VERSION = "17.1"`) did not change after v17.1. Every
version from v18 on is a standalone pre-registered experiment script
(`pretest_v*.py`, `score_prm_v22.py`) that is run on Kaggle or Colab.

## The system

```
problem
  -> Solver   (Qwen2.5-Math-7B-Instruct, 4-bit): k sampled CoT solutions, T=0.8
  -> Verifier (Qwen2.5-Math-PRM-7B, 4-bit): a reward for every step of every solution
  -> keep the solution whose LAST-step reward is highest          (best-of-k, "B3"/"B5")
```

The benchmark is GSM-Symbolic P2. Reasons:
- GSM-Hard has 16-21% defective golds (`gsmhard_audit.py`).
- GSM8K has no headroom left at 7B.

## Headline result: CONFIRMED on fresh rows

Pre-registered in the `pretest_v22.py` docstring. Evaluated on 100 FRESH P2
rows (seed 45, disjoint from the dev rows):

| arm | solver samples | accuracy |
|---|---|---|
| one sampled attempt (mean) | 1 | ~60% |
| S3 self-consistency | 3 | 66 |
| S5 self-consistency | 5 | 68 |
| **B3 verifier best-of-3** | 3 | **73** (W8 L1 vs S3) |
| **B5 verifier best-of-5** | 5 | **78** (W11 L1 vs S5, sign test p~0.006) |
| ADAPT: B3, plus 2 fresh samples only when gated (max-min step < 0.85) | 3.2 on average | 77 (exploratory, composed post hoc) |

Guard set (20 GSM-Plus distractor rows): 18/20 for every arm, so no harm.

## What did not work: pre-registered negative results

- **V2P (Verifier-to-Problem grounding)** is the thesis's original repair method. It maps the verifier's most-doubted step to the problem sentence behind it and re-solves with that sentence quoted.
  - Verdict: REFUTED.
  - Pooled over the dev pass and the confirmation run (25 gated rows): V2P 9, restart (2 fresh samples) 12, rewind 9.
  - Mechanism: a quoted clause gets re-read the same wrong way (p2_524: 14.8, 14.8 again), while fresh samples add diversity.
- **Rewind** (StepCo-style: continue from before the doubted step) also loses to restart (9 vs 12).
- **Phase 1 aggregation.** Min-over-steps (the pre-registered Phase 1 rule) gave only +1 (INCONCLUSIVE). Last-step aggregation was chosen on dev and then confirmed above.
- **v23, always-on V2P,** is implemented locally but NOT committed and NOT run. The confirmation above already shows fresh samples beating V2P samples at equal cost.

## Files

| file | what it is |
|---|---|
| `score_prm_v22.py` | Phase 1: PRM-scores stored traces (no generation) |
| `pretest_v22.py` | Phase 2: fresh-row confirmation, plus the `--dev` V2P pass and the `--pooled` reading |
| `pretest_data/v21_traces.json` | the 140 dev rows x 5 CoT samples (the v21 run) |
| `pretest_data/v22_dev_prm.json` | PRM step scores for those traces (Phase 1 output) |
| `pretest_data/v22_confirm_p2.json`, `v22_confirm_guard.json` | the fresh confirmation rows |
| `results_September/prm_scores_v22.json`, `v22_log.txt` | Phase 1 result and log |
| `results_September/pretest_v22_dev.json` | V2P dev pass (20 gated dev rows) |
| `results_September/preset_v22.json` | **confirmation run** (120 rows, all arms) |

Re-read the verdicts offline, with no GPU:

```bash
python score_prm_v22.py --analyse-only --out results_September/prm_scores_v22.json
python pretest_v22.py --summary-only --out results_September/preset_v22.json
python pretest_v22.py --pooled results_September/pretest_v22_dev.json results_September/preset_v22.json
```

## Next

**Update 2026-09-26:** one more accuracy experiment is planned, v23 (Verifier-guided Tabu
Restarts, an original repair method). See `V23_PLAN.md`. Its dev pass reuses the stored
v21/v22 drafts, so it needs about 6 GPU-hours and no new drafts.

Besides v23, the remaining work is writing.
- `paper_mas_sht.tex` is synced only through v15.3. It needs the fresh-seed null, v17, the GSM-Hard audit, v20 (SC baseline), v21 (error taxonomy: role errors), and v22.
- `references.bib` is missing from the repo; only `references_additions.bib` exists.
- The framing: an analysis of why MAS with small models fails (defective benchmarks, coupled agents, verifiers that certify difficulty rather than correctness, role errors), plus the confirmed Solver+Verifier system. The verifier method itself is not original (Lightman 2023, Qwen PRM). The contributions are the findings.
