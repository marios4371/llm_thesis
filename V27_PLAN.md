# v27: Null-Hypothesis Verification (dev screen plan, 2026-09-30)

v27 builds on the thesis's confirmed result, the v22 verifier (B5 78 vs SC@5
68 on fresh P2 rows). It fixes the blind spot we measured in it. The dev
screen generates no solutions and is pre-registered; it has not been run.

| file | what it is |
|---|---|
| `null_hypothesis.py` | the rule, the pools, the frozen bars |
| `pretest_v27.py` | the screen: writes the missing prototypes, scores every stored solution against its prototype, summarises |
| `test_v27.py` | 31 offline checks |
| `MAS_SHT_Kaggle_v27.ipynb` | Cell 1, then the tests, then the screen |

## 1. The blind spot, measured on stored data

**The verifier does not see recitation.** Take the 8 v22 rows where at
least 4 of 5 samples agree on a WRONG answer.
- It gives that answer a last-step reward of 1.000 on all 8.
- Its minimum step reward is 0.96-1.00 on 7 of them.
- Examples: "2 + 12 = 14 m per plant" when the 12 m already include the
  width; the non-rabbit white animals dropped from the total; the 60
  post-injury minutes applied before the injury.
- It catches only the arithmetic slip (min 0.32).

It shares the Solver's reading.

**A second reading does not help.** The Qwen Reader writes the same
"14 m per plant" and the same 37-minute return in its own notes.

**What does see the twist is the v26 Prototype agent.** Asked for the
familiar version of the problem, it deletes exactly "(including the plants
width)".

**Where the knowledge must NOT go: generation.** v26 showed that
contrasting or truncating generation kills the rare branches that carry the
right reading (−9.2 pp per sample).

## 2. The rule

1. Generate as usual. The pool is v25's RA: 2 plain samples plus 2 under the
   Reader's reading.
2. The verifier scores every candidate twice: under the real problem (as in
   v22) and under the prototype. The prototype is the null hypothesis: "this
   is the familiar problem".
3. **Among candidates tied at the top** (real reward within 0.001 of the
   best; 42 of 100 rows have two or more different answers there), choose
   the one the null scores LOWEST.

A recited solution is correct for the familiar problem, so the null scores
it high. A solution that applies the twist is wrong for the familiar problem,
so the null scores it low. The rule turns the verifier's own recitation bias
into a detector for it.

Safety, checked in `test_v27.py`:
- It only reorders candidates the verifier already rates as tied.
- With an identical or failed prototype, or equal null rewards, it IS v22.

## 3. Novelty (checked 2026-09-30)

**Not new:**
- PRM best-of-N.
- Counterfactual inputs as a test, in general:
  - Isomorphic Perturbation Testing (arXiv 2604.15149) uses ISOMORPHIC
    perturbations, to detect reward hacking in RL on logic induction;
  - a beam search over counterfactual contexts (arXiv 2609.37041) contrasts
    fixed "good"/"poor reasoning" templates.
- Same-model verifiers sharing the generator's blind spots is a known
  general observation.

**New, as far as the searches found:** scoring candidates against a
counterfactual *canonical version of the same problem*, written by an agent,
at inference time, to separate recited solutions from problem-specific ones.

## 4. Ceiling (stored data)

The null test can only help where the right answer is already inside the
top tie and v22's verifier picked a wrong one there.

| pool | v22 | ceiling |
|---|---|---|
| B5 | 78 | 81 |
| **RA** | **80** | **87** |
| RA5 | 81 | 88 |
| B10 | 81 | 90 |

That is why the primary pool is RA and not plain B5.

## 5. Pre-registration (frozen in `pretest_v27.py`)

**Screen** (100 dev main rows): NH-RA vs RA.
- GO if net ≥ +3 rows with W ≥ 2L.
- STOP if net ≤ 0.
- WEAK otherwise.

**Guard** (20 GSM-Plus rows): NH-RA − RA ≥ −1.

These rows motivated the idea, so GO only means "build the confirmation on
fresh rows". A fresh run must compare at equal calls: NH-RA uses 6 calls
(2 + Reader + 2 + prototype) against B6.

## 6. The workflow in plain words

1. Kaggle runs the tests (about 1 minute).
2. It writes the prototypes for the 42 rows that do not have one yet (about
   10 minutes).
3. It asks the verifier to re-score all the stored solutions against their
   prototypes (about 1 hour).
4. It prints GO / WEAK / STOP. You download `pretest_v27.json`, and I read
   it with `python pretest_v27.py --summary-only --out results_September/pretest_v27.json`.

## 7. Screen result (2026-09-30, `results_September/preset_V27_dev.json`): STOP

**NH-RA vs RA:** W4 L21, net −17, sign p = 0.001. NHV hurts on every pool:

| pool | v22 | NH |
|---|---|---|
| B5 | 78 | 61 |
| RA | 80 | 63 |
| RA5 | 81 | 62 |
| B10 | 81 | 62 |

- Guard: −1, which is within the bar.
- The prototypes worked: 120/120 usable, 31 identical.

**Why: the verifier barely reads the problem.**
- Its rewards under the prototype correlate r = 0.93 with its rewards under
  the real problem.
- The null reward is not a "twist detector". It is a weaker copy of the
  plain correctness signal: right candidates have a mean null reward of
  0.980, wrong ones 0.820.
- So the prediction runs backwards: AUROC 0.24 overall, 0.30 inside the
  RA top ties. Choosing the lowest null picks the weakest candidate in the
  tie.
- On the recitation rows the recited answer scores 1.000 under both
  problems, and so does the right answer where one exists (p2_890).

This is the known finding of Xu et al., "Reward Models Identify
Consistency, Not Causality" (arXiv 2502.14619): removing the problem
statement has minimal impact on reward scores. That paper was in the
novelty search results but was not read before building; it predicts this
failure.

**What it adds to the thesis.** It is the mechanism behind "the verifier is
blind to recitation". A process verifier checks the internal consistency of
the steps, not their fidelity to THIS problem. Swapping the problem for its
familiar version leaves its judgment almost unchanged (r = 0.93, measured on
69 rows x 10 candidates). No selection rule built on that verifier can
separate recited solutions from problem-specific ones.
