# v26: Prototype-Contrastive Decoding (pilot plan, 2026-09-29)

The v26 pilot tests whether the Solver can be made to reason about the
problem as written, instead of the familiar version it recites.

This changes each reasoning chain, not the selection among chains: no
verifier and no vote are involved. The pilot is pre-registered and has not
been run.

| file | what it is |
|---|---|
| `prototype_contrast.py` | the method: the Prototype agent's prompt and parser, the contrastive scorer, a decoding loop checked against HF `generate`, and the arms |
| `pretest_v26.py` | the pilot runner and its pre-registration (docstring), with resume and `--max-hours` |
| `test_v26.py` | 55 offline checks; `--require-model` fails if the model tests cannot run |
| `MAS_SHT_Kaggle_v26.ipynb` | Cell 1 of the Kaggle notebook, then tests, then a 2-row smoke run, then the pilot |

## 1. What goes wrong: a hand taxonomy

I read every row where none of the 5 plain samples is right: 18 of the 100
v22 rows (seed 45). Each was compared against GSM-Symbolic's
`original_question`, the GSM8K problem that P2 was built from by adding
clauses.

| type | rows | examples |
|---|---|---|
| **an explicit added or modified condition is overridden** | **12** | "(including the plants width)" but width + gap is used (3 rows); "these 3 groups", 2 counted (4); "16 white animals, half of which were rabbits", the other half dropped (2); post-injury "60 minutes on the beach" applied before the injury; an added return trip counted as one leg; 325 cal per serving read against a 325 g bag |
| gold or ambiguity | 4 | the gold floors fractional fruit (2); fees on the discounted price or not; an ill-posed signatures template |
| arithmetic slip | 2 | `198+92+594=984` on all 5 samples; `3+6+12+12+15=58` |

The errors sit on the clauses that P2 adds. A 2026 re-audit of GSM-Symbolic
(LessWrong, "Revisiting GSM-Symbolic") also reports that much of P2's drop
is ambiguity, which matches the 4 rows above.

## 2. The hole in the literature

Yan et al., *Recitation over Reasoning* (RoR-Bench, arXiv 2504.00509):
- **The finding.** Top models (o1, o3-mini, R1) lose about 60% when one
  phrase of a familiar problem changes.
- **What they tried** (quoted from the paper): "adding notice prompts and
  providing subtly modified problems as few-shots. Although these solutions
  can mitigate the performance drop slightly, they are far from
  satisfactory and a more complete solution is still yet to be proposed."
- **What made it worse:** showing the original problem in context.
- **Their Limitations:** "A more important and fundamental avenue for future
  research is to find an effective way for LLMs to overcome the problem of
  recitation over reasoning."

## 3. The method

1. **The Prototype agent** (Qwen2.5-7B-Instruct, greedy, 4 hand-written demos,
   none from the benchmark) rewrites the problem as its most familiar
   version: the same sentences, names, numbers and question, with the twist
   removed. This is an explicit "null hypothesis": the problem the Solver is
   likely to recite.
2. **The Solver** (Qwen2.5-Math-7B-Instruct) samples its chain for the real
   problem from
   `(1 + a) log p(y | real, y<t) - a log p(y | prototype, y<t)`.
   Only tokens with `p(y | real) >= b * max` are allowed. Both contexts share
   the generated prefix.
   - Where the texts agree, nothing changes.
   - Where the real text changes the situation, that change is amplified and
     the recited template is damped.
   - The constraint means PCD only re-ranks tokens the Solver already finds
     plausible for the real problem.
3. **The constants are the literature defaults.** a = 1.0 is CAD's setting
   for knowledge conflicts; b = 0.1 is Contrastive Decoding's plausibility
   threshold. Nothing is tuned on P2.

**Not new:** contrasting two conditional distributions.
- CAD (Shi et al., NAACL 2024) contrasts with vs without context, for
  summarization and knowledge-conflict QA.
- Contrastive Decoding (Li et al. 2023; O'Brien & Lewis 2023) contrasts an
  expert with an amateur model.
- Instructive decoding contrasts noisy instructions; source-contrastive MT
  contrasts a random other source.

**New, as far as the searches found:** the contrast input is a
counterfactual **canonical version of the same problem**, written by a
second agent, and it is used against recitation in multi-step reasoning.
That is the gap RoR-Bench names as open.

**The link to the thesis.** The prototype is the Solver's implicit
hypothesis about what the problem says. Decoding weighs the evidence of the
real text against that hypothesis, token by token. That is the "structured
hypothesis testing" of MAS-SHT, moved from candidate answers into the
reasoning itself.

## 4. Pilot design, pre-registered (the `pretest_v26.py` docstring)

**Rows.** The 100 v22 main rows. Stored samples 1-3 choose the rows, and
stored samples 4-5 are held out to check the sampler.
- hard: the 60 rows with at most 2 of samples 1-3 right;
- easy: 20 of the 40 rows with 3 of 3 right (seed 26), for the harm check.

**Arms** (k = 3 each, all drawn in one batch per row, v22 settings):

| arm | what it is |
|---|---|
| L0 | plain sampling (the sampler of every earlier version) |
| MINP | the plausibility constraint alone |
| **PCD** | a = 1.0 against the prototype |
| PCD5 | a = 0.5 (dose response) |
| CADQ | a = 1.0 against the question alone (a generic contrast) |

**Bars:**

| reading | rule |
|---|---|
| VALIDITY | \|L0 - stored samples 4-5\| ≤ 10 pp; otherwise no verdict |
| **PRIMARY** (hard rows) | PCD - L0 ≥ +5 pp and permutation p < 0.05 → SUPPORTED; ≤ 0 → REFUTED |
| ATTRIBUTE | PCD - MINP ≥ +3 pp |
| SPECIFIC | PCD - CADQ ≥ +3 pp |
| HARM (easy rows) | PCD - L0 ≥ -5 pp |
| GO | SUPPORTED and no harm → the fresh-row confirmation on the v25 seed-48 rows |

**Checked offline (test_v26.py):**
- On a tiny random Qwen2, the loop reproduces HF `generate` token for token
  (left padding, mixed lengths, early EOS).
- `warp()` equals HF's Temperature, TopK and TopP warpers.
- PCD against an identical prototype reproduces L0 exactly.
- Every PCD token passes the plausibility test on fresh forwards.
- Sampling is seeded and reproducible.
- An out-of-memory error falls back to drawing arm by arm.
- Qwen2.5-Math-7B-Instruct's generation_config sets no repetition penalty,
  so L0 equals the stored sampler. The runner warns if a config ever does.

## 5. Honest risks, written before the run

- **Amplifying an added clause is right when the Solver under-weights it**
  ("including the width"). It is wrong when the Solver over-applies it: in
  p2_890 the injury clause should NOT change the 60 minutes, and amplifying
  it may push toward the recited 170.
- **Counting errors may not respond.** Contrast re-weights tokens; it does
  not make the model count groups ("3 groups").
- **6 of the 18 rows are out of reach:** the 4 gold/ambiguity rows and the
  2 arithmetic slips.
- **Contrastive methods can derail long chains.** The b constraint and the
  easy-row HARM bar are there to catch that.

## 6. Run

1. Kaggle, T4 x2, Internet on, secret `HF_API_KEY`. Open
   `MAS_SHT_Kaggle_v26.ipynb` and use **Save & Run All**.
2. It runs the tests, then 2 smoke rows (read the first prototype and the
   s/row in the log), then the pilot, which stops itself at 7.6 h.
3. **Time.** Projected at 5-6 min per row, so 80 rows is about 7-8 h.
   - If it stops early, attach the Output and run again: it resumes.
   - A partial run gives no verdict.
4. Offline re-read:
   ```bash
   python pretest_v26.py --summary-only --out results_September/pretest_v26.json
   ```
