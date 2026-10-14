# v35: Trace-level SIV, symbolic refutation inside verifier-routed sampling

**Files:**
- `trace_siv.py`: the claim checker, the refutation policy and the rule (ADAPT-R).
- `replay_v35.py`: offline replay on every stored PRM-scored pool. No GPU.
- `test_v35.py`: 42 offline checks.

## 1. Where this comes from

Three measured facts were sitting in the stored pools.

1. **SIV's one reliable property is refutation.** When SIV rejects, it is right (P2 13/13, GSM-Hard 98.1%). A pass means little, and its bottleneck is the separate blueprint, which the 7B Architect gets right on only 13/48 P2 rows.
2. **The verifier's value flips with the error structure.** PRM best-of-5 beats SC@5 on P2 (+17) and loses to it on MATH-500 (−5). On MATH the PRM gives a last-step reward of 1.0 to wrong singletons.
3. **Those wrong MATH singletons are computational slips the PRM itself doubted.** The doubted step holds a provably false claim:

| row | claim in the doubted step | truth |
|---|---|---|
| math500_415 | `64^2 ≡ 64 (mod 11)` | 4 ≢ 9 |
| math500_272 | `9999 = 3·3·11·11·11` | = 11979 |
| math500_214 | `180+P+2P−30−75−75+15 → 180+3P−165+15` | the constant is −165, not −180 |

A Qwen-Math chain of thought is already an execution trace, because it writes its arithmetic out as LaTeX claims. So SIV's execution audit (Layer 1) can run on the trace itself. No blueprint and no extra call are needed, and the audit is used only to refute, never to certify.

## 2. The method

**The audit** is `trace_siv.audit(raw)`. It is pure Python plus SymPy, with no model call.
- Every PRM step is parsed for equality claims: LaTeX chains, plain `a + b = c`, congruences, and consecutive one-variable equations that share a numeric side.
- Each checkable claim gets TRUE or FALSE.
- Precision rules:
  - Words in `\text{}` become unit symbols. So `19 min + 18 min = 37 min` is checked, while `60 min = 1 hour` and `11 remainder 8` are not.
  - A unit on only one side can confirm a claim but never refute it.
  - The rounding a claim shows is tolerated.
  - Integer division, function notation, case splits and run-on chains are left unchecked.
  - Only numbers, single-letter names, unit symbols and operators ever reach SymPy's parser.

**The rule (ADAPT-R).** v22's gate and selection are unchanged; refuted samples are removed before the PRM picks.
1. Draw C1..C3 and audit them. Drop the refuted ones.
2. Take the best survivor by last-step reward.
3. If its min-step reward is at least 0.85, answer with it: 3 calls.
4. Otherwise draw C4 and C5, audit them, and answer with the best survivor of the five: 5 calls.
5. If all five are refuted, use v22's pick over all five.

With nothing refuted, ADAPT-R equals `gated_reading.ARMS['ADAPT5']` on every stored v22 and MATH row (test_v35 part 4).

## 3. The refutation policy: `trace_siv.refutes(verdicts, rewards)`

**Decided and frozen 2026-10-14: "any FALSE claim"**, before any v35 sample existed.

This is the method's precision/recall knob. ADAPT-R was measured under each candidate on all pools:

| policy | P2 (200) | MATH (100) | V21 (100, untouched) | guard (80) | right samples refuted P2 / MATH / V21 | wrong samples refuted P2 / MATH / V21 |
|---|---|---|---|---|---|---|
| any FALSE claim | **160** | **94** | **86** | 70 | 2 / 11 / 5 | **80** / 14 / **38** |
| FALSE in a step with reward < 0.85 | 157 | 94 | 86 | 70 | 2 / 8 / 4 | 44 / 12 / 31 |
| FALSE in a step with reward < 0.5 | 158 | 92 | 85 | 70 | 2 / 7 / 3 | 30 / 7 / 26 |
| ≥ 2 FALSE claims | 158 | 93 | 85 | 70 | 1 / 5 / 0 | 3 / 9 / 6 |
| FALSE in the last 3 steps | 158 | 91 | 85 | 70 | 0 / 4 / 1 | 30 / 8 / 11 |
| (no refutation = v22 ADAPT) | 158 | 90 | 85 | 70 | – | – |

**What to weigh:**
- **Precision.** All 6 FALSE verdicts inside right V21 samples are genuinely false claims the model later recovered from. A refuted right sample is therefore a real recovered slip, not a parser error.
- **Recall.** The "doubted step" policies lose the correlated slips. Those are scored 0.99–1.00 by the PRM, which is exactly the case only the audit can see.
- **The invariants** in test_v35 part 3 hold for every policy above.

## 4. Offline evidence (`python replay_v35.py`, policy "any FALSE")

| arm | P2 (200) | MATH (100) | V21 (100) | guard (80) | calls/row |
|---|---|---|---|---|---|
| SC@5 | 142 | **95** | 80 | 67 | 5 |
| B5 (PRM best-of-5) | 159 | 90 | 86 | 69 | 5 |
| ADAPT (v22) | 158 | 90 | 85 | 70 | 3.37 |
| **ADAPT-R** | **160** | 94 | **86** | **70** | **3.37** |

**Paired comparisons:**
- **P2:** ADAPT-R vs SC@5 is W22 L4 (p = 0.0005); vs ADAPT it is W2 L0.
- **MATH:** vs ADAPT it is W4 L0; vs SC@5 it is W1 L2.
- **V21:** vs SC@5 it is W7 L1; vs ADAPT it is W1 L0.
- ADAPT-R never loses a row to ADAPT in 480 rows.

**Mechanism numbers:**
- **The PRM misses a third of the false steps.** It scores ≥ 0.9 on 32/86 provably false P2 steps and 11/38 MATH steps.
- **Correlated slips.** Example: `61 + 8 + 13 + 10 + 58 = 140` appears in 4/5 samples of p2_185, scored 0.99–1.00. Of SC@5's 58 P2 failures, 14 have a refutable wrong majority, and 7 of those share the same false claim.
- **Taxonomy of selection-relevant errors.** Scattered slips are majority-safe, but the PRM saturates on them, so the audit is needed. Correlated slips fool both the vote and the PRM, and the audit catches them. Correlated misreadings (P2's recitation, 44/58) fool all three; nothing here fixes them.

**Caveat.** P2a, P2b and MATH are the rows the audit's false positives were read and fixed on, so they are in-sample for the checker. V21 is the only untouched pool, and there the gain is +1. The MATH gain (+4) needs fresh rows.

## 5. Novelty check (2026-10-13, papers read, not titles)

**Not new:**
- **T1** (Kang et al., ICLR 2026, arXiv 2504.04718) filters with a tool and then lets a PRM or GenRM pick, which is the same two-stage form (their Eq. 3).
  - Its ToolV is an LLM writing verification code per candidate. It is distilled from a teacher and costs about 2× compute.
  - Its own appendix says it does not capture "misinterpreting problem structure".
- **MATH-VF** (arXiv 2505.20869) uses an LLM Formalizer plus a CAS/SMT Critic.
- **Themis** (ICLR 2024) is a tool-augmented reward model.

**Not found anywhere:**
1. An LLM-free audit of the trace's own claims: zero calls, no training, deterministic.
2. Correlated arithmetic slips across independent samples that a same-policy PRM scores ~1.0. "When Self-Consistency Backfires" (2608.11403) asks for "a signal external to the model's own samples". "Decomposing Wrong-Consensus Agreement" (2608.18795) does not classify causes.
3. The error-structure account of when PRM selection beats voting, together with refutation removing that dependence.
4. Composition with verifier-routed compute (ADAPT).

The honest label is **partially original**: original audit and findings, known two-stage form.

## 6. Confirmation run (FROZEN 2026-10-14, before any sample exists)

**Rows.** `pretest_data/v35_math500.json`: 80 MATH-500 test problems with a plain numeric answer, none of v34's 100, **levels 3–5 only**, seed 35 (23 L3, 36 L4, 21 L5, drawn from 154 eligible rows).
- The levels were chosen from metadata, never from outcomes.
- All four rows where refutation changed v34's answer were level 3–5, and levels 1–2 cost half the time for no signal.
- So the claim's scope is "MATH-500, levels 3–5".

**Samples.** Exactly as v34: k = 5, T 0.8, the same prompt, 2048 tokens and the same PRM. Every arm is read off the same samples.
- The GPU part does not depend on the refutation policy, because the audit runs offline on the saved traces.
- `trace_siv.refutes()` and the checker are committed before the run. The summary prints their sha256, and it refuses to read a verdict while `refutes()` is unwritten.

**Bars:**

| reading | bar |
|---|---|
| PRIMARY | ADAPT-R vs ADAPT: net ≥ +3 with W ≥ 2L → SUPPORTED; net ≤ 0 → REFUTED; otherwise INCONCLUSIVE |
| SECONDARY | ADAPT-R vs SC@5: net ≥ −2 → NON-INFERIOR |
| PRECISION | every FALSE verdict inside a right sample is printed and hand-checked |

**How to run (two machines in parallel):**
- **Kaggle** (`MAS_SHT_Kaggle_v35.ipynb`, T4 x2) runs `--reverse`, from the end of the list, and stops after 2.3 h to fit a 3 h quota. That is about 28 rows at about 300 s/row.
- **Colab** (`MAS_SHT_Colab_v35.ipynb`, one T4) runs from the start. It saves to `MyDrive/MAS_SHT_v35/` after every row; "Run all" again resumes.
- Upload Kaggle's `pretest_v35.json` to `MyDrive/MAS_SHT_v35/kaggle_pretest_v35.json`. Colab's Cell 4 merges the two files, and a row complete in either counts once.
- Locally: `python pretest_v35.py --summary-only --out A.json --merge B.json`.

**Cost.** About 300 s per level 3–5 row on Kaggle, so about 6.7 GPU-hours in total. Colab's single T4 is somewhat slower.

## 7. Thesis framing

SIV is the original mechanism. Its measured profile (refutations right, passes uninformative, blueprint the bottleneck) dictated its final role. In that role it is a free, deterministic refuter applied to the Solver's own trace, composed with verifier-routed sampling. The three-class error taxonomy (scattered slips / correlated slips / correlated misreadings) explains when each component helps, and it names the residual (recitation) that no same-family component fixes.
