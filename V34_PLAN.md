# v34: Verifier-routed sampling on MATH-500 (generalization)

**Files:**
- `pretest_v34.py`: the manifest, the draws and the reading.
- `test_v34.py`: 23 offline checks.
- `pretest_data/v34_math500.json`: 100 rows, seed 34.
- `MAS_SHT_Kaggle_v34.ipynb` and `MAS_SHT_Colab_v34.ipynb`.
- The figure of the P2 result is in `figures/compute_accuracy.{png,pdf}` (`python figures/plot_compute_accuracy.py`).

## 1. What is being generalized

**The rule** is v22's ADAPT, frozen since v22 (`gated_reading.ARMS['ADAPT5']`):
1. Draw 3 samples and score them with the PRM.
2. If the minimum step reward of the best one (by last-step reward) is below 0.85, draw 2 more.
3. Answer with the best last-step reward.

It costs 3 or 5 calls.

**What it showed on GSM-Symbolic P2:** 200 fresh rows from two independent sets (v22 confirmation, seed 45; v25 confirmation, seed 48).

| arm | right / 200 | calls/row |
|---|---|---|
| SC@5 | 142 | 5 |
| B5 (verifier best-of-5) | 159 | 5 |
| **ADAPT5** | **158** | **3.33** |

- ADAPT5 vs SC@5: **+16, W20 L4, p = 0.0015.** The paired bootstrap 95% CI of the difference is [+3.5, +13.0] pp.
- **The threshold barely matters:** 0.5 → 77.5%, 0.85 → 79.0%, 0.99 → 80.0%. A threshold sweep is descriptive; 0.85 is the pre-registered value.

## 2. Why MATH-500

| property | GSM-Symbolic P2 | MATH-500 |
|---|---|---|
| difficulty | grade-school arithmetic | harder, levels 1–5 |
| content | word problems | algebra, number theory, geometry, counting, precalculus |
| match with the PRM's training | further | closer |

A positive result turns "it works on P2" into "it works across benchmarks".

## 3. Rows and samples

**Rows:**
- 100 of the 317 MATH-500 test problems whose answer is a plain number. Seed 34; every level 1–5 is present.
- Answers are read from the last `\boxed{}` that holds a number. `\frac`, `°` and `%` are handled. The 9 parser cases are in test_v34.py.

**Samples:**
- 5 per row from Qwen2.5-Math-7B-Instruct, with v22's sampler and CoT prompt.
- Up to **2048 tokens**, because MATH solutions are longer than GSM's.
- All are scored by Qwen2.5-Math-PRM-7B.

## 4. Pre-registered (frozen 2026-10-12, before any sample exists)

| reading | bar |
|---|---|
| **PRIMARY 1** | ADAPT5 vs SC@5. **SUPPORTED** if net ≥ +5 with W ≥ 2L and sign p < 0.05; **REFUTED** if net ≤ 0; otherwise INCONCLUSIVE |
| **PRIMARY 2** | ADAPT5 vs B5. **NON-INFERIOR** if net ≥ 0 at ≤ 4.0 calls/row; **BEATS** if net ≥ +3 with W ≥ 2L; otherwise NOT MET |

Reported, not decisive: S3, B3, accuracy by level, the gate rate, per-sample accuracy and unparsed answers.

## 5. Cost and how to run

- **Kaggle** (`MAS_SHT_Kaggle_v34.ipynb`, T4 x2, Save & Run All): about 5–6 min per problem, so about 9–10 h, i.e. two sessions.
  - For the second session, add the first session's Output as an input.
- **Reading the result:** `python pretest_v34.py --summary-only --out results_Octomber/pretest_v34.json`.

## 6. Risks, stated before the run

1. **The gate was set on GSM-like rows.** On MATH the PRM's step rewards may sit lower, so it may gate more often: more calls, which tests PRIMARY 2's ≤ 4.0 bound. Or it may sit higher and gate less. Either outcome is a fair test.
2. **The comparison may be saturated.** Qwen2.5-Math-7B-Instruct is strong on easy MATH. If SC@5 is already about 85%, the margin for +5 is small. Levels 4–5 carry the headroom.
3. **Truncation.** A sample cut at 2048 tokens gives no answer. The unparsed count is reported.

## 7. Result

Not run yet.
