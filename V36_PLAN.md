# v36: the large run of Trace-level SIV, on rows where it can act

**Files:**
- `pretest_v36.py`: the manifest, the run and the reading. It is a thin wrapper over v34's runner.
- `test_v36.py`: 30 offline checks.
- `pretest_data/v36_arith.json`: 200 rows, seed 36.
- `MAS_SHT_Kaggle_v36.ipynb` runs forward for 8.5 h. `MAS_SHT_Colab_v36.ipynb` runs `--reverse` and saves to Drive.
- `trace_siv.py` is now v35.1: the v35 checker plus two precision fixes.

## 1. Why a new run

v35 (fresh MATH L3-5) ended INCONCLUSIVE: W1 L0 against ADAPT. The cause was the rows, not the audit. Only 5 of 80 rows were selection opportunities, i.e. ADAPT was wrong while the pool held a right sample.

Refutation acts only on such rows, and only when the wrong pick holds a checkable false claim. On the stored P2 pools, the share of wrong samples carrying a provably false claim grows with the size of the numbers in the problem text:

| max number in the text | < 100 | 100–999 | 1e3–1e4 | ≥ 1e4 |
|---|---|---|---|---|
| rows | 88 | 87 | 17 | 8 |
| per-sample accuracy | 0.72 | 0.57 | 0.69 | 0.20 |
| wrong samples with a FALSE claim | 13% | 25% | 19% | **41%** |

GSM-Hard's own failure mode is "last-digit arithmetic slips" and carrying errors at large magnitude (the magnitude and depth findings). This is exactly what a CAS check of the trace's claims refutes.

## 2. Rows (built 2026-10-16 from metadata only, before any sample)

| pool | n | source | filter | eligible |
|---|---|---|---|---|
| **A** | 100 | GSM-Hard | passes `gsmhard_audit.py` (auditable, none of collision / never perturbed / Platinum-dropped), gold > 0, max number in the text ≥ 1e4, none of the v18/v19 pretest rows | 619 |
| **B** | 50 | GSM-Plus "digit expansion" (human-verified golds) | numeric answer, max number in the text ≥ 1e3 | 487 |
| **C** | 50 | GSM-Plus "integer-decimal-fraction conversion" | numeric answer | 1317 |

- The median of the largest number in the text is about 5.8 million in A, 5,600 in B and 20 in C. C is the decimal and fraction pool: decimal-addition slips such as `4.32 + 4.95 + 64.80 = 73.07` are what the audit catches.
- Rows are interleaved A, B, A, C, …. Each half of the two-machine run therefore covers all three pools.
- The checker was never developed on any of these datasets. Its v35.1 fixes were found on the v35 MATH rows, which are already spent.

## 3. Samples

Exactly v22/v34, so every arm reads the same samples:
- 5 samples from Qwen2.5-Math-7B-Instruct with v22's CoT prompt, T 0.8.
- **1536** new tokens, because large-number arithmetic runs longer than P2.
- Every sample is scored by Qwen2.5-Math-PRM-7B.
- The audit runs offline on the saved traces.

## 4. Pre-registered (FROZEN 2026-10-16, before any sample exists)

| reading | bar |
|---|---|
| **PRIMARY** | ADAPT-R vs ADAPT, all 200 rows. **SUPPORTED** if net ≥ +5 with W ≥ 2L, labelled "significant" if the sign test gives p < 0.05. **REFUTED** if net ≤ 0. Otherwise INCONCLUSIVE. |
| **SECONDARY** | ADAPT-R vs SC@5, all 200 rows: net ≥ 0 → NON-INFERIOR |
| **PRECISION** | every FALSE verdict inside a right sample is printed and hand-checked |

**Reported, not decisive:**
- every arm per pool;
- B5-R vs B5 and SC5-R vs SC5;
- the number of selection opportunities;
- calls/row and the audit statistics;
- the fresh total pooled with v35 (280 rows).

## 5. Cost and how to run

- About 250 s per row, from v22/v35 on Kaggle and Colab. 200 rows is about **14 GPU-hours**.
- **Kaggle** (`MAS_SHT_Kaggle_v36.ipynb`, T4 x2, Internet on, secret `HF_API_KEY`): Save & Run All. It works from the start, stops itself after 8.5 h (about 120 rows), and saves after every row.
- **Colab** (`MAS_SHT_Colab_v36.ipynb`, T4, secret `HF_API_KEY`): Run all. It works from the end (`--reverse`) and saves to `MyDrive/MAS_SHT_v36/` after every row. If the session dies, run all again.
- **When Kaggle finishes:** download `pretest_v36.json` and upload it to Drive as `MyDrive/MAS_SHT_v36/kaggle_pretest_v36.json`. Colab's Cell 4 then gives the reading.
- **Locally:** `python pretest_v36.py --summary-only --out A.json --merge B.json`.
- If the two halves have not met when Kaggle stops, a second Kaggle session resumes it (Add Input → the first version's Output).

## 6. What each outcome would mean

- **SUPPORTED.** Trace-level SIV improves verifier-routed sampling where computation is the bottleneck. This would be the fresh confirmation that v35 could not give for lack of opportunities.
- **INCONCLUSIVE with 0 losses.** The audit is safe but small. Report the opportunity count next to the result, as in v35.
- **REFUTED.** The audit does not pay off even where errors are arithmetic. The PRM would then already catch most slips on these rows; the audit statistics will show it (PRM ≥ 0.9 on false steps).
