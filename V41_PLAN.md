# v41: v38 replicated on fresh rows — ADAPT-RS vs ADAPT-R, same code, new data

**Files:**
- `pretest_v41.py`: swaps v38's manifest and output for v41's and reads the result. The run itself is `pretest_v38.run`, unchanged.
- `test_v41.py`: checks that the audit, the surgery file, the policy and the bars are v38's frozen ones, and that no row was seen before.
- `pretest_data/v41_replication.json` (sha256 **ba6b2e38d01b**): 319 rows, seed 41, built 2026-10-24 from metadata only.
- `MAS_SHT_Kaggle_v41.ipynb`: T4 x2, 8.5 h per session; it resumes from an uploaded `pretest_v41.json`.
- Frozen and unchanged from v38: `trace_siv.py` **483ed23b89b6**, `trace_surgery.py` **42376931d94a**, `surgery_policy()` **8eb4df69f91b**.

## 1. Why

v38 is the thesis's strongest result. On 200 fresh clean GSM-Hard rows, ADAPT-RS beat ADAPT-R 131 to 122 (W9 L0, p = 0.004). It also beat ADAPT W15 L0 and SC@5 W13 L2.

Nine wins is a small count on one set of rows. v41 runs **the identical system** on every remaining eligible row. That replicates the effect on data the method never saw, and lets the paper report it pooled over v38 + v41, about 450–520 rows.

## 2. Rows

GSM-Hard rows that meet all of the following:
- auditable and clean under `gsmhard_audit.py` (v38's three defect channels);
- gold > 0;
- largest number in the text ≥ 10,000;
- in no earlier manifest: v38, v36, the v18/v19 rows (and therefore not v40's either).

That leaves **319 rows**, shuffled with seed 41. The run takes them in order and stops by time, never by outcome.

## 3. Pre-registered (FROZEN 2026-10-24, before any v41 sample exists)

Canonical grader |pred − gold| < 1e-3. The analysis set is the complete rows, and a verdict needs **≥ 150** of them.

- **PRIMARY.** ADAPT-RS vs ADAPT-R, with v38's bars.
  - **REPLICATED:** net ≥ +3 with W ≥ 2L. It is "significant" if the sign test gives p < 0.05.
  - **NOT REPLICATED:** net ≤ 0.
  - **INCONCLUSIVE:** anything else.
- **SECONDARY.** ADAPT-RS vs SC5, self-consistency at 5 calls. Net ≥ 0 means **NO WORSE THAN THE VOTE**.
- **Reported, not decisive:**
  - v38 + v41 pooled, for both comparisons;
  - ADAPT-RS vs ADAPT and the fixed policies;
  - coverage, resume diagnostics, calls per row.

Applied to v38's own 200 rows, this reading prints *REPLICATED and significant: W9 L0* (`test_v41.py`, part 3).

## 4. Cost and schedule (Kaggle's 30 h)

v38 took about 4 minutes per row on T4 x2, so one 8.5-hour session does about 120 rows.

| session | notebook | what | hours |
|---|---|---|---|
| 1 | `MAS_SHT_Kaggle_v41.ipynb` | v41 rows 1–~120 | 8.5 |
| 2 | `MAS_SHT_Kaggle_v41.ipynb` (Input: session 1's `pretest_v41.json`) | v41 rows ~120–~245 | 8.5 |
| 3 | `MAS_SHT_Kaggle_v40.ipynb` | v40: DeepSeek-Math-7B and Granite-3.1-8B, one per GPU, the 200 v38 rows | 8.5 |
| 4 | `MAS_SHT_Kaggle_v40_prm.ipynb` (Input: every `pretest_v40_*.json`, laptop and Kaggle) | PRM scores for v40 | ~2–3 |
| (5) | `MAS_SHT_Kaggle_v41.ipynb` | the remaining v41 rows, only if the quota allows | ≤ 8.5 |

Sessions 1–4 total about 28 h. After session 2, v41 already has about 245 rows, above the 150 a verdict needs.

## 5. Reading

    python pretest_v41.py --summary-only --out results_Octomber/pretest_v41.json

The v38 files in `results_Octomber/` add the pooled v38 + v41 line.

## 6. RESULT (2026-10-29; 254 of 319 rows; Kaggle sessions 1 and 3)

`results_Octomber/preset_V41_2nd.json` holds rows 0–253 of the manifest, in order. Its first 121 rows are byte-identical to session 1's file. The frozen hashes are v38's, and no surgery-stage errors were recorded. Session 2 did not resume (its input file had another name) and re-drew rows 0–117. Its numbers are kept below as a re-draw; the file itself was overwritten.

| arm | right / 254 | calls/row |
|---|---|---|
| SC5 | 161 | 5.00 |
| ADAPT | 165 | 3.44 |
| ADAPT-R | 169 | 3.53 |
| **ADAPT-RS** | **177** | 3.49 |

**The pre-registered reading** (≥ 150 rows):
- **PRIMARY: REPLICATED and significant.** ADAPT-RS vs ADAPT-R: W9 L1, net +8, sign p = 0.0215.
- **SECONDARY: NO WORSE THAN THE VOTE.** ADAPT-RS vs SC5: W20 L4, net +16, p = 0.0015.

**Reported, not decisive:**
- **v38 + v41 pooled, 454 rows.** ADAPT-RS vs ADAPT-R: **W18 L1, net +17, p = 0.0001**. ADAPT-RS vs SC5: W33 L6, net +27, p < 0.0001.
- **Surgery.** 143 refuted samples, all of them operable. Propagation rescued 99 of 140 wrong samples and broke 1 of 3 right ones. Coverage: 11 of the 75 rows with no right sample gain one.
- **Precision of the audit.** It refuted 3 of 731 right samples and 140 of 378 wrong ones.
- **Session 2's independent re-draw of rows 0–117.** ADAPT-RS 79, ADAPT-R 77, ADAPT 73, SC5 73. ADAPT-RS vs ADAPT-R: W2 L0. Vs SC5: W6 L0.
- **The remaining 65 rows (254–318)** can still be run (Colab); the verdict above does not depend on them.
