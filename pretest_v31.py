"""
[v31.0] Dev screen: Cross-Family Tie Jury (XJ). Where v22's verifier ties
different answers, does a model from another family (DeepSeek-Math-7B-RL),
solving the problem on its own, pick the right one more often than v22?

The rule and the literature are in cross_family_jury.py and V31_PLAN.md.
No Qwen model and no verifier is loaded: the Qwen samples and v22's step
rewards are the stored ones. The only new generation is the juror's k = 5
solutions, and only on the rows that need them.
  dev rows   results_September/preset_V25_dev.json (100 main + 20 guard,
             C5 + Q5 with v22's rewards, seed 45)
  held-out   pretest_data/v21_traces.json + v22_dev_prm.json (seed 44,
             100 main + 40 guard, C5 only), read by no version from v24 on

Offline (test_v31.py): v22's rule gives B10 81 / B5 78 on dev, B5 86 on the
held-out rows. A perfect juror would give B10 90, B5 81, held-out B5 90.

PRE-REGISTERED, frozen 2026-10-06 before any juror sample exists
---------------------------------------------------------------
  SCREEN (100 dev main rows)  XJ-B10 vs B10 (v22's rule, same 10 samples):
                              net >= +3 with W >= 2L -> GO (fresh confirmation);
                              net <= 0 -> STOP;  otherwise WEAK
  GUARD  (20 GSM-Plus rows)   XJ-B10 - B10 >= -1
  HELD   (secondary)          the 100 held-out main rows: XJ-B5 - B5 >= 0 ->
                              CONSISTENT, else INCONSISTENT
  Rule: xj_pick in cross_family_jury.py (juror plurality among the tied
  answers; no agreement or a juror tie keeps v22's pick).
Reported, not decisive: XJ on B5/RA/RA5; the juror's own accuracy; the
SHARED-ERROR FLOOR (on rows where >= 3 of Qwen's 5 plain samples agree on a
wrong answer, how often the juror's plurality is that same wrong answer); the
EXPLORATORY Jury-10 control (plain vote over Qwen C5 + juror 5, the "LLMs as
a Jury" recipe without the PRM); a second juror family (OLMo-2-7B) if run.

MODES
-----
    python pretest_v31.py --max-hours 4.0         # Kaggle T4 x2 (or Colab T4); ~2 h to the verdict
    python pretest_v31.py --summary-only          # re-read, no GPU
    python pretest_v31.py --stub                  # offline plumbing, no GPU
    python pretest_v31.py --second-juror ...      # also OLMo-2 on the tie rows (exploratory)
Re-running the same command resumes (every row is saved as it is sampled).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional, Sequence

import cross_family_jury as XJ
import pretest_v21 as V21
import pretest_v24 as V24
import pretest_v30 as T30
import reread_verification as RV

OUT = 'pretest_v31.json'
OUT_STUB = 'pretest_v31_stub.json'
POOL_ORDER = ('B10', 'RA', 'RA5', 'B5')
STAGES = ('pass 1 (dev rows with a contested B10 tie)',
          'pass 1b (held-out rows with a contested B5 tie)',
          'pass 2 (every other dev main row: Jury-10 control and the shared-error floor)',
          'pass 3 (second juror on the dev tie rows, exploratory)')


# ---------------------------------------------------------------------------
# rows and passes
# ---------------------------------------------------------------------------

def load_all(args) -> List[Dict]:
    return T30.load_rows(args.v25, args.v26) + T30.load_held_rows(args.held_traces, args.held_prm)


def needs_jury(row: Dict, stage: str) -> Optional[str]:
    """The juror this row needs in `stage`, or None."""
    held = T30.is_held(row)
    tie = bool(RV.contested_tie(RV.candidates(row, 'B5' if held else 'B10')))
    if stage == STAGES[0]:
        return XJ.JUROR if (not held and tie) else None
    if stage == STAGES[1]:
        return XJ.JUROR if (held and tie) else None
    if stage == STAGES[2]:
        return XJ.JUROR if (row['set'] == 'main' and not tie) else None
    if stage == STAGES[3]:
        return XJ.JUROR_2 if (not held and tie) else None
    raise ValueError(stage)


def missing(row: Dict, rec: Dict, stage: str) -> Optional[str]:
    j = needs_jury(row, stage)
    return j if j and XJ.votes_of(rec, j) is None else None


def primary_complete(rows: Sequence[Dict], recs: Dict) -> bool:
    return all(missing(r, recs.get(r['key'], {}), STAGES[0]) is None for r in rows)


# ---------------------------------------------------------------------------
# the juror
# ---------------------------------------------------------------------------

FALLBACK_TEMPLATE = ("{% for m in messages %}{% if m['role'] == 'user' %}User: {{ m['content'] }}\n\n"
                     "{% else %}{{ m['content'] }}\n\n{% endif %}{% endfor %}"
                     "{% if add_generation_prompt %}Assistant:{% endif %}")


class _StubJuror:
    """Offline: right with p = 0.5, otherwise a scattered wrong answer."""
    provider = 'stub'

    def __init__(self, rows: Sequence[Dict], name: str):
        self.gold = {r['text']: r['gold'] for r in rows}
        self.rng = random.Random(len(name))

    def solve(self, text: str, k: int) -> List[str]:
        g = self.gold.get(text, 0.0)
        return [f"Reasoning.\nAnswer: {g if self.rng.random() < 0.5 else g + self.rng.randint(1, 9):g}"
                for _ in range(k)]


class LocalJuror:
    def __init__(self, name: str):
        import torch
        from Mas_solver import UnifiedLLMClient
        self.name = name
        self.client = UnifiedLLMClient(provider='local_hf', use_cache=False, model_override=name,
                                       load_4bit=True, device_index=0 if torch.cuda.is_available() else None)
        self.client._ensure_local_model()
        tok = self.client._local_tokenizer
        if not getattr(tok, 'chat_template', None):
            tok.chat_template = FALLBACK_TEMPLATE
            print(f"  {name}: no chat template, using a plain User/Assistant one", flush=True)
        print(f"  juror loaded: {name}", flush=True)

    def solve(self, text: str, k: int) -> List[str]:
        return [str(x) for x in V21.sample(self.client, V21.COT_PROMPT.format(problem=text), k,
                                           XJ.TEMPERATURE, XJ.MAX_TOKENS)]


def build_juror(args, rows, name):
    return _StubJuror(rows, name) if args.stub else LocalJuror(name)


def free(juror) -> None:
    import gc
    if juror is not None and hasattr(juror, 'client'):
        juror.client._local_model = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# reading the results
# ---------------------------------------------------------------------------

def juror_report(rows: Sequence[Dict], recs: Dict, juror: str) -> List[str]:
    acc, n, rows_any = 0, 0, 0
    for r in rows:
        v = XJ.votes_of(recs.get(r['key'], {}), juror)
        if v is None:
            continue
        n += len(v)
        acc += sum(V21.correct(x, r['gold']) for x in v)
        rows_any += any(V21.correct(x, r['gold']) for x in v)
    if not n:
        return []
    return [f"{juror.split('/')[-1]}: {100 * acc / n:.1f}% of its samples right on the "
            f"{n // XJ.K_JUROR} rows it solved; some sample right on {rows_any}"]


def shared_error_floor(rows: Sequence[Dict], recs: Dict, juror: str = XJ.JUROR) -> List[str]:
    """On rows where >= 3 of Qwen's 5 plain samples agree on a WRONG answer (a
    self-consistent error), does the juror's plurality repeat it?"""
    same = other_wrong = right = none = n = 0
    for r in rows:
        v = XJ.votes_of(recs.get(r['key'], {}), juror)
        if v is None:
            continue
        qa = [s.get('answer') for s in r['C'][:5]]
        groups = [g for g in V21_groups(qa) if len(g) >= 3]
        if not groups or V21.correct(qa[groups[0][0]], r['gold']):
            continue
        n += 1
        wrong = qa[groups[0][0]]
        jp = XJ.plain_vote(v)
        if jp is None:
            none += 1
        elif V21.correct(jp, r['gold']):
            right += 1
        elif XJ.agree(jp, wrong):
            same += 1
        else:
            other_wrong += 1
    if not n:
        return []
    return [f"shared-error floor: {n} rows where >= 3/5 Qwen samples agree on a wrong answer; the juror's "
            f"plurality repeats that answer on {same} ({100 * same / n:.0f}%), is right on {right}, "
            f"is a different wrong answer on {other_wrong}, empty on {none}"]


def V21_groups(answers):
    import score_prm_v22 as P
    return P.L.clusters(list(answers))


def dispute_report(rows: Sequence[Dict], recs: Dict, pool: str) -> List[str]:
    with_right = v22 = xj = 0
    why = collections.Counter()
    for r in rows:
        cands = RV.candidates(r, pool)
        tie = RV.contested_tie(cands)
        if not tie or not any(V21.correct(cands[i]['answer'], r['gold']) for i in tie):
            continue
        with_right += 1
        votes = XJ.votes_of(recs.get(r['key'], {}))
        v22 += V21.correct(XJ.choose(r, pool, None), r['gold'])
        xj += V21.correct(XJ.choose(r, pool, votes), r['gold'])
        why[XJ.xj_pick(cands, votes)[1]] += 1
    if not with_right:
        return []
    return [f"{pool}: {with_right} contested ties contain the right answer; right after the tie: "
            f"v22 {v22} ({100 * v22 / with_right:.0f}%), XJ {xj} ({100 * xj / with_right:.0f}%); "
            f"decided by: {dict(why)}"]


def summarise(rows: List[Dict], recs: Dict, stub: bool = False) -> int:
    held = [r for r in rows if r['set'] == 'held']
    held_all = [r for r in rows if T30.is_held(r)]
    dev = [r for r in rows if not T30.is_held(r)]
    main = [r for r in dev if r['set'] == 'main']
    guard = [r for r in dev if r['set'] == 'guard']
    print('\n' + '=' * 78)
    done = primary_complete(dev, recs)
    print(f"  v31 Cross-Family Tie Jury (juror {XJ.JUROR}): pre-registered reading "
          f"{'complete' if done else 'PARTIAL'}")
    print(f"\n    {'pool':5s} {'v22':>4s} {'XJ':>4s} {'W':>3s} {'L':>3s}   (100 dev main rows)")
    for pool in POOL_ORDER:
        s = XJ.paired(main, recs, pool)
        print(f"    {pool:5s} {s['base']:4d} {s['xj']:4d} {s['w']:3d} {s['l']:3d}")
    s2 = XJ.paired(main, recs, 'B10', juror=XJ.JUROR_2)
    if any(XJ.votes_of(recs.get(r['key'], {}), XJ.JUROR_2) for r in main):
        print(f"    second juror ({XJ.JUROR_2.split('/')[-1]}), B10: {s2['xj']} vs {s2['base']} "
              f"W={s2['w']} L={s2['l']}  (exploratory)")
    for line in dispute_report(main, recs, 'B10') + dispute_report(main, recs, 'B5'):
        print('  ' + line)
    for line in (juror_report(main, recs, XJ.JUROR) + juror_report(main, recs, XJ.JUROR_2)
                 + shared_error_floor(main, recs)):
        print('  ' + line)
    jv = [XJ.votes_of(recs.get(r['key'], {})) for r in main]
    if all(v is not None for v in jv):
        j10 = sum(V21.correct(XJ.plain_vote([s.get('answer') for s in r['C'][:5]] + v), r['gold'])
                  for r, v in zip(main, jv))
        s5 = sum(V21.correct(XJ.plain_vote([s.get('answer') for s in r['C'][:5]]), r['gold']) for r in main)
        print(f"  EXPLORATORY Jury-10 (plain vote, Qwen C5 + juror 5, no PRM): {j10}  vs  Qwen SC@5 {s5}")
    if held:
        h = XJ.paired(held, recs, 'B5')
        print(f"\n  held-out v21 rows ({len(held)} main, B5): XJ-B5 {h['xj']} vs B5 {h['base']}, "
              f"W={h['w']} L={h['l']}")
        for line in dispute_report(held, recs, 'B5') + shared_error_floor(held, recs):
            print('  ' + line)
    c = XJ.paired(main, recs, XJ.PRIMARY_POOL)
    if c['wins'] or c['losses']:
        print(f"  flips on B10: wrong->right {c['wins']}; right->wrong {c['losses']}")
    print('\n  pre-registered reading:')
    if not done:
        print("    none: PARTIAL run (pass 1 incomplete). Re-run the same command.")
        return 0
    for v in XJ.screen_verdict(main, guard, recs):
        print('    ' + v)
    if held and all(missing(r, recs.get(r['key'], {}), STAGES[1]) is None for r in held_all):
        print('    ' + XJ.held_verdict(held, recs))
    return 0


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def _save(path: str, recs: Dict, meta: Dict) -> None:
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(dict(meta, rows=recs), fh, ensure_ascii=False)
    os.replace(tmp, path)


def run(args) -> int:
    rows = load_all(args)
    out_path = args.out or (OUT_STUB if args.stub else OUT)
    recs: Dict[str, Dict] = {}
    if os.path.exists(out_path) and args.resume:
        recs = V24._read(out_path).get('rows', {})
        print(f"resuming: {len(recs)} rows in {out_path}")
    for r in rows:
        recs.setdefault(r['key'], {'jurors': {}})
        recs[r['key']].setdefault('jurors', {})
    meta = {'juror': XJ.JUROR, 'juror_2': XJ.JUROR_2, 'k': XJ.K_JUROR, 'temperature': XJ.TEMPERATURE,
            'max_tokens': XJ.MAX_TOKENS, 'prompt': V21.COT_PROMPT, 'eps': RV.EPS}
    stages = list(STAGES[:3]) + ([STAGES[3]] if args.second_juror else [])
    print(f"rows: {len(rows)} ({sum(not T30.is_held(r) for r in rows)} dev, "
          f"{sum(T30.is_held(r) for r in rows)} held-out)", flush=True)
    deadline = time.time() + args.max_hours * 3600 if args.max_hours else 0.0
    jurors: Dict[str, object] = {}
    t0, made = time.time(), 0
    for stage in stages:
        todo = [r for r in rows if missing(r, recs[r['key']], stage)]
        if not todo:
            continue
        print(f"\n{stage}: {len(todo)} rows", flush=True)
        for i, row in enumerate(todo, 1):
            name = missing(row, recs[row['key']], stage)
            if deadline and time.time() > deadline:
                _save(out_path, recs, meta)
                print("\n  --max-hours reached; re-run the identical command to resume.")
                return summarise(rows, recs, args.stub)
            if name not in jurors:
                for other in list(jurors):
                    free(jurors.pop(other))
                jurors[name] = build_juror(args, rows, name)
            ts = time.time()
            raws = jurors[name].solve(row['text'], XJ.K_JUROR)
            recs[row['key']]['jurors'][name] = [{'raw': x[:3000], 'answer': V21.parse_answer(x)} for x in raws]
            made += 1
            _save(out_path, recs, meta)
            held = T30.is_held(row)
            pool = 'B5' if held else 'B10'
            votes = XJ.votes_of(recs[row['key']], name)
            b = V21.correct(XJ.choose(row, pool, None), row['gold'])
            a = V21.correct(XJ.choose(row, pool, votes), row['gold'])
            print(f"[{i:3d}/{len(todo)}] {row['set']:10s} {row['pid'][:24]:24s} juror right "
                  f"{sum(V21.correct(v, row['gold']) for v in votes)}/{len(votes)}  {pool}: "
                  f"v22 {'R' if b else '-'} XJ {'R' if a else '-'}  {time.time() - ts:5.0f}s", flush=True)
    _save(out_path, recs, meta)
    print(f"\n  saved: {out_path}")
    return summarise(rows, recs, args.stub)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--v25', default=T30.DEV_V25)
    ap.add_argument('--v26', default=T30.DEV_V26)
    ap.add_argument('--held-traces', default=T30.HELD_TRACES)
    ap.add_argument('--held-prm', default=T30.HELD_PRM)
    ap.add_argument('--out', default='')
    ap.add_argument('--stub', action='store_true')
    ap.add_argument('--no-resume', dest='resume', action='store_false')
    ap.add_argument('--max-hours', type=float, default=0.0)
    ap.add_argument('--second-juror', action='store_true')
    ap.add_argument('--summary-only', action='store_true')
    args = ap.parse_args()
    if args.summary_only:
        path = args.out or (OUT_STUB if args.stub else OUT)
        return summarise(load_all(args), V24._read(path).get('rows', {}), args.stub)
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
