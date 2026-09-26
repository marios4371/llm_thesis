# v23: Verifier-guided Tabu Restarts (plan as of 2026-09-26)

This is the next accuracy experiment. It follows `V22_RESULTS.md`. The
method is original to this thesis. It is built on the confirmed
Solver+Verifier system, and it is pre-registered and not yet run.

| file | what it is |
|---|---|
| `tabu_restart.py` | the method: leader, trigger, doubted step, negative-evidence notes. Deterministic, no model |
| `pretest_v23.py` | the pre-registered runner: dev pass on stored rows, fresh confirmation, summary |
| `v23_diagnosis.py` | the offline evidence that motivates the method (no GPU) |
| `test_v23.py` | 61 offline checks (`python test_v23.py`) |

## 1. Why: where the remaining errors are

Every number in this section comes from `python v23_diagnosis.py`. It reads
the traces already on disk, with no GPU.

| rows | C1 (1 sample) | S5 | B5 | oracle@5 | B5 errors with no right sample |
|---|---|---|---|---|---|
| v21 dev P2 (100) | 69 | 80 | 86 | 90 | 10 of 14 |
| v22 fresh P2 (100) | 56 | 68 | **78** | 82 | **18 of 22** |

- **Selection is spent.** B5 is 4 rows from oracle@5 on both sets. No new
  vote, weighting or verifier can add much. The errors that remain are
  rows where no sample is right, so the gain has to come from producing
  a better sample.
- **The shared error is a step, not an answer.** On the 32 triggered rows
  whose leader is wrong (pooled):
  - a fresh sample is right only **16%** of the time;
  - it repeats the leader's wrong *answer* only **22%** of the time;
  - it re-derives the value of the *step the Verifier doubts* **43%** of
    the time.
- **Choosing is not the bottleneck; producing is.**
  - A right sample outscores a wrong leader **58%** of the time.
  - A wrong sample outscores a right leader only **7%** of the time.
- **Three repairs have already been tried, and none beat blind restarts**
  (pooled v22 gated rows):

  | repair | what the Solver was given | correct rows |
  |---|---|---|
  | V2P | a quote of the implicated sentence (positive attention) | 9 |
  | RW | its own prefix up to the doubted step | 9 |
  | RST | nothing | 12 |

## 2. The method

```
problem ─► Solver: 3 sampled CoT drafts
        ─► Verifier (PRM): a reward for every step
        ─► leader = draft with the highest last-step reward        (= B3)
           leader's lowest step < TAU (0.95)?  no ─► output the leader
                                                yes ▼
        ─► Tabu keeper (deterministic): the leader's lowest-rewarded computing step
        ─► Solver ×2, FRESH context: problem + "this step was checked and is WRONG:
           «step». Do not repeat that mistake; solve again from the beginning."
        ─► Verifier scores the restarts against the ORIGINAL problem
        ─► output = best of {3 drafts + 2 restarts} by last-step reward
```

This is **tabu search** (Glover 1986) moved into the space of readings of
a word problem:

| tabu search | here |
|---|---|
| objective function | the step verifier |
| neighbourhood | fresh samples |
| tabu list | the steps the verifier rejected, delivered as negative evidence |
| aspiration criterion | the original drafts stay in the pool, so a tabu reading can still win on score |

**The design rule is about what information flows between agents.** The
Solver:
- never sees a failed solution (the anchoring that sank RW and
  self-correction);
- never gets a positive cue to re-read (what sank V2P);
- is not blind (what limits RST).

It gets exactly one verifier-localised negative fact. `test_v23.py`
checks this on all 80 triggered rows:
- the note holds the doubted step and no other step;
- a restart prompt is the RST prompt plus the note, byte for byte.

### What is new, and what is not

**Not new.** These are used as components or as baselines:
- PRM best-of-N (Lightman 2023; Qwen PRM);
- rewind from the doubted step (StepCo, ACL 2025);
- self-refinement with the failed solution in context (Self-Refine,
  Reflexion);
- hints built from earlier *answers* (Progressive-Hint Prompting);
- invalid demonstrations drawn from *other* problems (Contrastive CoT);
- independent verification questions (CoVe, SelfCheck).

**New in v23:**
1. The negative evidence is one step of the same problem, chosen by a
   learned step verifier.
2. It goes to a fresh context and never comes with the failed solution.
3. The procedure is framed and tested as tabu search over readings. The
   verifier is the objective and the drafts are the aspiration pool.
4. The ANS control keeps the same frame and gives only the leader's final
   answer as the negative fact. This isolates whether the verifier's
   *localisation* is what matters.

**Before writing related work, read these.** They were found by search
and not read here:
- "Reason, Reward, Refine" (arXiv 2607.05199): step-level corrections
  with structured feedback for small models, in physics;
- "Hint Marginalization" (OpenReview).

Neither appeared to restart in a fresh context with verifier-localised
negative evidence. That still has to be checked.

## 3. Pre-registration (frozen in `pretest_v23.py` before any TABU sample exists)

**Arms.** Every repair arm spends the same 2 extra samples, and only on
triggered rows. On any other row it is B3.

| arm | what it is |
|---|---|
| C1 | one CoT sample (the simple prompt) |
| S3, S5 | self-consistency |
| B3, B5 | verifier best-of-N (v22) |
| RST | blind restarts (the v22 winner) |
| **TABU** | **VTR** |
| ANS | answer-only exclusion, the attribution control |
| MIX | 1 RST + 1 TABU, post hoc |
| TABU2, V2P, RSTN | optional, via `--arms`: tabu with memory, v22's quote, same-session blind restarts |

**Dev pass** (`--dev`). It runs on the stored drafts of the v21 rows (140)
and the v22 rows (120), so no new drafts are drawn. TAU was fixed on the
v21 rows only (it catches 15/20 wrong leaders there), and the v22 rows
are held out for it. The pre-registered reading is on the pooled main
rows (n=200, 63 triggered, 28 with a wrong leader):

| condition | reading |
|---|---|
| TABU − RST ≥ +3 rows and wins ≥ 2× losses | **SUPPORTED**: run the fresh confirmation |
| TABU ≤ RST | **REFUTED** |
| anything else | INCONCLUSIVE |
| TABU − ANS ≥ +2 | the verifier's localisation is what helps |
| ANS ≥ TABU | excluding the answer is enough |
| guard rows (n=60): TABU − B3 ≥ −1 | no harm |

**Fresh confirmation** (100 new P2 rows, seed 46, plus 20 guard rows).
The same TABU vs RST rule applies, and in addition:
**TABU − S5 ≥ +4 rows with wins ≥ 2× losses** means the full system beats
self-consistency@5 while drawing fewer samples (3 + 2·trigger rate ≈ 3.6
instead of 5).

**What it takes to pass (a rough estimate, not part of the rule).** On
the stored rows, RST beats B3 by W7 L3 on main. The estimate below uses
the displacement rates of section 1. To clear RST by +3:
- TABU restarts need to be right about **28–30%** of the time on
  wrong-leader rows, against 16% for blind restarts;
- at the same time they must not produce high-scoring wrong answers on
  the 35 triggered rows whose leader is right.

The sample-level table the runner prints measures exactly this.

## 4. How to run (Kaggle T4×2 or Colab, the v22 setup)

```bash
python test_v23.py                                   # offline, 1 min
python v23_diagnosis.py                              # offline, the numbers above

# 1) dev pass on stored rows: 80 triggered rows x 2 arms x 2 samples, ~6 h
python pretest_v23.py --dev --max-hours 8.0
#    (TABU only, ~3.5 h:  python pretest_v23.py --dev --arms TABU --max-hours 8.0
#     then add ANS later with --arms TABU,ANS; the TABU samples are kept)
python pretest_v23.py --dev --summary-only           # re-read, no GPU

# 2) only if the dev pass reads SUPPORTED: fresh confirmation
python pretest_v23.py --build-manifest               # once, needs `datasets`; commit the 2 files
python pretest_v23.py --max-hours 8.0                # ~9 h in total: two sessions, it resumes
python pretest_v23.py --summary-only
```

Time estimate, from v22's measured seconds per row:
- a row with 5 samples + PRM takes ~194 s;
- 2 arms × 2 samples + PRM add ~245 s to a triggered row.

Download `pretest_v23_dev.json` (or `pretest_v23.json`) after every
session. A stub run (`--stub`) writes `pretest_v23_stub.json`, never the
real files.

## 5. How either outcome goes into the thesis

- **SUPPORTED.** An original repair that turns the verifier's step
  rejection into a better sample. It would be the thesis's method
  contribution, on top of the confirmed Solver+Verifier system.
- **REFUTED.** The same table still completes the story of repairs:
  - positive evidence (V2P) < blind restarts;
  - negative evidence (TABU), measured against both;
  - the ANS control says whether localisation mattered at all.

  Together with v14 (verifiers certify difficulty) and v21 (role errors),
  that is a mechanism-level account of why small models do not repair
  their own readings.
