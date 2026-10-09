# v31: Cross-Family Tie Jury (XJ), an extension of the v22 system

**Files:** `cross_family_jury.py` (the rule; pure), `pretest_v31.py` (the dev screen), `test_v31.py` (24 offline checks, including that every notebook cell compiles), `MAS_SHT_Kaggle_v31.ipynb`, `MAS_SHT_Colab_v31.ipynb`.

The base is v22 (Qwen2.5-Math-7B-Instruct samples, best last-step reward of Qwen2.5-Math-PRM-7B): B5 78 vs SC@5 68 on fresh rows, W11 L1. XJ changes nothing in v22 except **who decides the rows where v22's verifier cannot**.

## 1. What five screens established

| version | same-family agent asked to check problem fidelity | result |
|---|---|---|
| v24 | a Reader writes the reading for the Solver | the Solver adopts it both ways (+19.6 pp misread rows, −12.5 pp well-read) |
| v27 | the PRM scores each solution under the familiar problem | PRM problem-insensitive, r = 0.93 |
| v28 | DPO against the familiar-version solution | −4.5 pp held-out templates |
| v29 | a Questioner + factored answers settle the dispute | 47% unusable questions; answers right 74% < v22's 84% |
| v30 | the PRM re-reads the problem after the solution | B10 73 vs 81; AUROC in ties 0.77 → 0.62 |

Every agent there shares the generator's reading. The stored Reader notes show the mechanism concretely: on p2_734 the Reader writes *"Not specified, but it is the same as … 42 minutes"* and all 5 note-conditioned samples are wrong; on p2_1972 it writes *"600 minutes"* and 4/5 are right.

## 2. The published ideas and their open windows

- **Tan et al., EMNLP 2025, "Too Consistent to Detect"** (arXiv 2505.17656). Self-consistent errors are **model-specific**: cross-family overlap 5.4–20.4% (Qwen2.5-7B vs 14B: 28.7%). They only **detect** them, with a probe trained on a second model's hidden states. Verbatim: *"The underlying causes of consistent errors still require deeper investigation"*; mitigation is left to *"the community"*.
- **"LLMs as a Jury"** (arXiv 2607.10139). Independently trained models' wrong answers scatter while the right one accumulates agreement, so cross-model consensus can beat four trained verifiers. Its stated ceiling is a **shared-error floor** where models share a misconception. It never studies recitation of perturbed problems, where every family has memorised the same GSM8K originals.
- **Thesis (v14 certificate–difficulty, v22 ADAPT):** verification works as **routing**, not arbitration.

**XJ's increment:**
- **Correction**, not detection, of self-consistent errors, with no training and no hidden states.
- Heterogeneity spent **only inside the PRM's measured blind spot** (its contested ties), so the cost is 5 juror samples on about 60% of rows.
- A direct **measurement of the shared-error floor on recitation**: when ≥ 3 of Qwen's 5 samples agree on a wrong P2 answer, how often does a different family repeat it?

Not new: cross-model voting, PRM best-of-n, model cascades.

## 3. The method

1. v22 as stored: B10 = 5 plain + 5 Reader-view Qwen samples, last-step PRM reward.
2. If the PRM's top (within 1e-3) holds two or more answers, the **juror** (DeepSeek-Math-7B-RL: other lab, other pre-training corpus, RL-trained, 4-bit) solves the problem **5 times** at T = 0.8 with the same CoT prompt. It never sees a candidate, so it cannot adopt a reading (v24's failure), and it solves the whole problem, so nobody has to localise the dispute (v29's failure).
3. Each tied answer scores the number of juror answers that agree with it. The highest score wins. No agreement, or a juror tie, keeps v22's pick.

## 4. Ceiling (offline, test_v31.py)

| pool | v22 | perfect juror |
|---|---|---|
| B5 | 78 | 81 |
| RA | 80 | 87 |
| RA5 | 81 | 88 |
| **B10** | **81** | **90** |
| held-out B5 (v21 seed-44 rows) | 86 | 90 |

## 5. Pre-registered (frozen 2026-10-06, before any juror sample exists)

| | rule |
|---|---|
| **SCREEN** (100 dev main rows) | XJ-B10 vs B10 (v22's rule, same samples): net ≥ +3 with W ≥ 2L → **GO**; net ≤ 0 → **STOP**; otherwise WEAK |
| GUARD (20 GSM-Plus rows) | XJ-B10 − B10 ≥ −1 |
| HELD (secondary) | the 100 held-out v21 main rows (B5 only): XJ-B5 − B5 ≥ 0 → CONSISTENT, else INCONSISTENT |
| rule | `xj_pick`: juror plurality among the tied answers; 0 agreement or a juror tie → v22's pick; juror k = 5, T = 0.8, DeepSeek-Math-7B-RL |

**Reported, not decisive:** XJ on B5/RA/RA5; the juror's own accuracy; the shared-error floor (dev and held-out); the EXPLORATORY Jury-10 control (plain vote over Qwen C5 + juror 5, the Jury paper's recipe without a PRM); a second family (OLMo-2-7B-Instruct) on the dev tie rows; every flip.

On GO, the confirmation is the fresh seed-48 rows (`pretest_data/v25_confirm_p2.json`): C5 + Reader + Q5 + PRM + juror on the tie rows, about 12 h.

## 6. Cost and how to run

- Juror samples only where needed: pass 1 = 72 dev tie rows (64 main + 8 guard), pass 1b = 52 held-out tie rows. About **2–3 h to the verdict** on a T4.
- Pass 2 = the other 36 dev main rows (controls), pass 3 = the second juror (`--second-juror`).
- **Kaggle** (recommended): `MAS_SHT_Kaggle_v31.ipynb`, GPU T4 x2, Save & Run All, `--max-hours 6.0`. **Colab:** `MAS_SHT_Colab_v31.ipynb`, results on Drive, resumable.
- Offline reading: `python pretest_v31.py --summary-only --out <file>`.

## 7. Risks, stated before the run

- **Shared-error floor.** If DeepSeek-Math has memorised GSM8K the same way, it recites the same wrong answers and the jury moves nothing. The floor line measures exactly this; a STOP with a high floor is itself the answer to the Jury paper's open ceiling on recitation.
- **A weaker juror.** It only has to prefer the right *tied* answer over the wrong one. Its own accuracy can be below Qwen's, as long as its wrong answers scatter.
- **Dev rows read by the error analysis.** Hence the held-out reading and, on GO, the fresh confirmation.

## 8. Screen result (2026-10-08, Kaggle, `results_Octomber/preset_V31_dev.json`): STOP

| | v22 | XJ | W | L |
|---|---|---|---|---|
| **B10 (pre-registered)** | **81** | **75** | 3 | 9 |
| B5 | 78 | 75 | 1 | 4 |
| held-out B5 | 86 | 82 | 2 | 6 |
| guard | | ±0 | 2 | 2 |

- **SCREEN STOP** (net −6, sign p = 0.146); **HELD INCONSISTENT**. Ties right: XJ 42/57 (74%) vs v22 48/57 (84%); held-out 26/34 vs 30/34.
- **The juror is too weak to arbitrate:** DeepSeek-Math-7B-RL gets 44.8% of its samples right (some sample right on 69/100 rows), against roughly 60% for Qwen. The second family is weaker still (OLMo-2-7B: 21.2%, exploratory, 16 rows).
- **But the errors ARE decorrelated: the shared-error floor on recitation is low.** On the 13 dev rows where ≥ 3/5 Qwen samples agree on a wrong answer, the juror's plurality repeats that answer on 1 (8%); it is right on 5 and a *different* wrong answer on 7. Held-out: 1/7 (14%). This matches Tan et al.'s 5–20% cross-family overlap and answers the Jury paper's open ceiling for this error type: P2 recitation errors are model-specific, not a shared misconception.
- EXPLORATORY Jury-10 (plain vote over Qwen C5 + juror 5, no PRM): 76 vs Qwen SC@5 68 (+8), still below v22's B5 78.
- Post hoc (EXPLORATORY, not claims): overriding only on ≥ 3/5 juror votes gives dev B10 82 / B5 79 but held-out B5 85 vs 86; adding Qwen tie counts gives 82 / 77 / 84. Nothing survives the held-out rows.

**What it adds to the thesis.** Heterogeneity supplies exactly the decorrelation the five same-family screens lacked (shared-error floor 8–14%), but a cross-family juror helps only if it is roughly as accurate as the generator. Inside v22's ties the PRM is already right 84–91%, so a 45%-accurate juror loses more ties than it wins. The lever is real; its price is a juror of comparable strength.
