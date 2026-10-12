# v33: Verifier-Gated Reading Arbitration (VGRA)

**Files:**
- `gated_reading.py`: the rule, the comparators and the verdicts; pure code.
- `pretest_v33.py`: drives `pretest_v25.py`'s confirmation mode, unchanged, then reads the result.
- `test_v33.py`: 18 offline checks.
- `MAS_SHT_Kaggle_v33.ipynb` and `MAS_SHT_Colab_v33.ipynb`.

## 1. Why: build on what worked

v26–v32 all chased one residual: the ~9 reading disputes inside the PRM's saturated top tie. That slice has a ceiling of +9 rows and ±4 rows of noise at n = 100. Meanwhile three components that **worked** were never combined:

| component | measured |
|---|---|
| v22 Verifier (Qwen2.5-Math-PRM-7B, best last-step reward) | B5 78 vs SC@5 68 on fresh rows, W11 L1 (confirmed) |
| v24 Reader (Qwen2.5-7B-Instruct, sentence-anchored notes) | the Solver adopts the reading: +19.6 pp per sample on misread rows, but −12.5 pp on well-read rows |
| v22 ADAPT gate (min step reward of the best of 3 plain samples < 0.85) | finds the rows where the verifier doubts; defined on the v21 rows |

VGRA calls the Reader **only where the verifier doubts**. That keeps the Reader's gain on misread rows and avoids its harm on well-read ones. The verifier then arbitrates between the two readings.

## 2. Novelty (checked 2026-10-11)

**Not new:**
- PRM best-of-n.
- Uncertainty-gated test-time compute. Entropy-Gated Branching, uncertainty-aware budget allocation, calibrated-PRM instance-adaptive scaling and BETAPRM all spend **more samples or branches** where the model or the verifier is unsure.
- Rephrasing the problem: RaR is always on and uses no verifier.

**New:**
- On verifier doubt, the compute goes to a **different agent that re-reads the problem**.
- The verifier then **arbitrates between readings**.
- This turns the v24 Reader, measured as harmful when always on, into a routed agent.

## 3. The rule

1. Draw C1..C3 with the plain CoT prompt and v22's sampler (T 0.8, top-p 0.95, top-k 50, 1024 tokens), and score them with the PRM.
2. Let b be the best of C1..C3 by last-step reward; ties go to the lower position.
3. If min(step rewards of b) ≥ 0.85, the answer is b's answer. This costs **3 calls**.
4. Otherwise, call the Reader once and draw Q1, Q2 under its reading. Score them against the ORIGINAL text. Answer with v22's rule over {C1, C2, C3, Q1, Q2}. This costs **6 calls**.

## 4. Dev numbers: in-sample, NOT the claim

These are the stored v22 seed-45 rows. The gate and the Reader were fixed before these rows existed, but the composition was chosen after looking at them.

| arm | right / 100 | calls/row |
|---|---|---|
| SC@5 | 68 | 5 |
| B3 | 73 | 3 |
| ADAPT5 (same gate, 2 more **plain** samples) | 76 | 3.32 |
| B5 (v22) | 78 | 5 |
| RA (always-on reading, v25) | 80 | 5 |
| **VGRA** | **80** | **3.48** |

- **Comparisons:**
  - VGRA vs SC@5: W13 L1, p = 0.002.
  - VGRA vs B5: W6 L4.
  - VGRA vs ADAPT5: W5 L1. The Reader matters, not just extra samples.
- **On the 16 gated rows:** B3 gets 4 right, B5 7, ADAPT5 7, VGRA 11.
- **Guard:** 17 vs B5 18.

## 5. Pre-registered (frozen 2026-10-11, before any confirmation sample exists)

**Rows:** the v25 confirmation rows, never run.
- 100 GSM-Symbolic P2 rows and 20 GSM-Plus distractor rows, seed 48.
- They are disjoint from every row of v20–v32.

**Samples:** `pretest_v25.py`'s confirmation mode, unchanged.
- 5 plain samples, one Reader call and 2 read samples per row, all scored.
- VGRA is charged the Reader and the read samples only on the rows it gates.

| reading | bar |
|---|---|
| **PRIMARY 1** (main) | VGRA vs SC@5. **SUPPORTED** if net ≥ +5 with W ≥ 2L and sign p < 0.05; **REFUTED** if net ≤ 0; otherwise INCONCLUSIVE |
| **PRIMARY 2** (main) | VGRA vs B5 (v22). **BEATS** if net ≥ +3 with W ≥ 2L at ≤ 4.0 calls/row; **NON-INFERIOR** if net ≥ 0 at ≤ 4.0 calls/row; otherwise NOT MET |
| GUARD (20 rows) | VGRA − B5 ≥ −1 |

**Reported, not decisive:**
- the arms B3, S3, RA and ADAPT5;
- accuracy on the gated rows;
- pooled dev + confirmation, which must always be labelled as such.

## 6. Cost and how to run

- **Kaggle** (`MAS_SHT_Kaggle_v33.ipynb`, GPU T4 x2): about 11 h in total, so two sessions.
  - Each session stops by itself at 8 h.
  - For the second session, add the first session's Output as an input and run again; it resumes.
- **Colab** (`MAS_SHT_Colab_v33.ipynb`): the result is kept on Drive. Expect several sessions.
- **Reading the result:** `python pretest_v33.py --summary-only --out results_Octomber/pretest_v25.json`.

## 7. Risks, stated before the run

1. **Few gated rows.** The gate fired on 16% of dev rows. If fresh rows are easier, VGRA ≈ B3 and PRIMARY 2 rests on B3 vs B5.
   - On dev, B3 lost 5 rows to B5 on ungated rows. The Reader won them back on gated rows.
2. **Small n.** At 100 rows, PRIMARY 2's non-inferiority (net ≥ 0) is a weak bar. Its value is at 30% less compute.
   - PRIMARY 1 (vs SC) has the power: the dev effect is +12 against a bar of +5.
3. **A Reader failure.** If the Reader writes nothing usable, gated rows fall back to plain samples (VGRA ≈ ADAPT). `pretest_v25.py` aborts if its first 5 readings are unusable.

## 8. Result: FINAL (2026-10-12, 100 main + 20 guard, `results_Octomber/pretest_v25_2ndsession.json`)

**Pre-registered reading:**
- **PRIMARY 1, INCONCLUSIVE:** VGRA 78 vs SC@5 74, W7 L3, net +4, p = 0.34 (bar: +5 and p < 0.05).
- **PRIMARY 2, NOT MET:** VGRA 78 vs B5 81, W2 L5, at 3.51 calls/row.
- **GUARD, no harm:** −1.

**The Reader routing did not replicate.**
- On the 17 gated rows the Reader view gets 6 right, while two more plain samples get 10.
- The dev effect (11 vs 7 on 16 rows) was in-sample.

**What DID replicate** (reported as a replication; not a v33 primary):
- The v22 ADAPT rule is fixed since v22: when the min step reward of the best of 3 samples is below 0.85, it adds C4 and C5.

| set | ADAPT5 | B5 | SC@5 | ADAPT5 calls/row |
|---|---|---|---|---|
| v25 confirmation (100) | 82 | 81 | 74 | 3.34 |
| v22 confirmation (100) | 76 | 78 | 68 | 3.32 |
| **pooled, 200 fresh P2 rows** | **158** | **159** | **142** | **3.33** |

**Pooled comparisons:**
- **ADAPT5 vs SC@5: +16, W20 L4, p = 0.0015.**
- ADAPT5 vs B5: −1 (W2 L3) at 67% of B5's compute.
- ADAPT5 vs B3: +8 (W9 L1, p = 0.02).
- B5 vs SC@5: +17 (p = 0.0009).
- Guard, 40 rows: all arms score 35.
