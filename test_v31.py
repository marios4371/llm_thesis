"""[v31.0] Guards for the Cross-Family Tie Jury screen. Offline, no GPU.

What these tests exist to prevent is a screen that measures a bug:
  * a comparator that is not v22's rule (B10 81 / B5 78 dev, B5 86 held-out);
  * a jury that reaches outside the PRM's contested tie, or that moves a pick
    on no evidence (no agreement, a juror tie, missing votes);
  * juror samples spent on rows the verdict does not need, or a verdict read
    off a partial run;
  * a notebook cell that does not even compile (the v30 Kaggle run died on
    one), or bars that moved after freezing.

Run as `python test_v31.py`.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile

import cross_family_jury as XJ
import pretest_v21 as V21
import pretest_v24 as V24
import pretest_v30 as T30
import pretest_v31 as T
import reread_verification as RV

FAILS = []
N = [0]


def check(cond, label):
    N[0] += 1
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        FAILS.append(label)


def _args(**kw):
    base = dict(v25=T30.DEV_V25, v26=T30.DEV_V26, held_traces=T30.HELD_TRACES, held_prm=T30.HELD_PRM,
                out='', stub=True, resume=True, max_hours=0.0, second_juror=False, summary_only=False)
    base.update(kw)
    return argparse.Namespace(**base)


ROWS = T.load_all(_args())
DEV = [r for r in ROWS if not T30.is_held(r)]
MAIN = [r for r in DEV if r['set'] == 'main']
GUARD = [r for r in DEV if r['set'] == 'guard']
HELD = [r for r in ROWS if r['set'] == 'held']


def _acc(rows, pool, votes_fn):
    return sum(V21.correct(XJ.choose(r, pool, votes_fn(r)), r['gold']) for r in rows)


def part1():
    print("\nPART 1 - the comparator and the rule")
    check((len(MAIN), len(GUARD), len(HELD)) == (100, 20, 100), "100 dev main + 20 guard + 100 held-out main rows")
    got = {p: _acc(MAIN, p, lambda r: None) for p in RV.POOLS}
    check(got == {'B5': 78, 'RA': 80, 'RA5': 81, 'B10': 81}, f"v22 reproduced on dev (got {got})")
    check(_acc(HELD, 'B5', lambda r: None) == 86, "v22 reproduced on the held-out rows: B5 86")
    gold = lambda r: [r['gold']] * 5
    ceil = {p: _acc(MAIN, p, gold) for p in RV.POOLS}
    check(ceil == {'B5': 81, 'RA': 87, 'RA5': 88, 'B10': 90}, f"a perfect juror gives the tie ceiling (got {ceil})")
    check(_acc(HELD, 'B5', gold) == 90, "a perfect juror gives 90 on the held-out B5 pool")
    same = all(XJ.choose(r, p, []) == XJ.choose(r, p, None) for r in ROWS for p in ('B5', 'B10')
               if not (T30.is_held(r) and p == 'B10'))
    check(same, "an empty vote list keeps v22's pick everywhere")
    off = all(XJ.choose(r, 'B10', [-12345.0] * 5) == XJ.choose(r, 'B10', None) for r in DEV)
    check(off, "votes for an answer outside the tie never move the pick")
    inside = True
    for r in DEV:
        c = RV.candidates(r, 'B10')
        tie = RV.contested_tie(c)
        for a in {c[i]['answer'] for i in range(len(c)) if c[i]['answer'] is not None}:
            i, _ = XJ.xj_pick(c, [a] * 5)
            inside &= (i in tie) if tie else (i == XJ.xj_pick(c, None)[0])
    check(inside, "whatever the votes, the pick stays inside the contested tie")
    # a juror tie between two tied answers keeps v22
    r = next(x for x in MAIN if x['pid'].endswith('_1268'))
    c = RV.candidates(r, 'B10')
    tie = RV.contested_tie(c)
    answers = []
    for i in tie:
        if not any(XJ.agree(c[i]['answer'], a) for a in answers):
            answers.append(c[i]['answer'])
    check(len(answers) >= 2 and XJ.xj_pick(c, [answers[0], answers[1]])[1] == 'juror tie'
          and XJ.xj_pick(c, [answers[0], answers[1]])[0] == XJ.xj_pick(c, None)[0],
          "one vote each for two tied answers -> juror tie -> v22's pick")
    check(XJ.choose(r, 'B10', [19.0, 19.0, 16.0]) == 19.0 and XJ.choose(r, 'B10', None) == 16.0,
          "p2_1268: two votes for 19 against one for 16 flip v22's 16 to 19")
    check(XJ.agree(19.0, 19.0004) and not XJ.agree(19.0, 20.0) and not XJ.agree(None, 19.0),
          "answers agree within the repo's grading tolerance")


def part2():
    print("\nPART 2 - what gets sampled")
    need = {s: [r for r in ROWS if T.needs_jury(r, s)] for s in T.STAGES}
    check(len([r for r in need[T.STAGES[0]] if r['set'] == 'main']) == 64,
          f"pass 1 samples the 64 dev main rows with a contested B10 tie (+{sum(r['set'] == 'guard' for r in need[T.STAGES[0]])} guard)")
    check(all(T30.is_held(r) for r in need[T.STAGES[1]]) and
          sum(r['set'] == 'held' for r in need[T.STAGES[1]]) == 41,
          f"pass 1b samples the 41 held-out main rows with a contested B5 tie (+{sum(r['set'] == 'held-guard' for r in need[T.STAGES[1]])} guard)")
    check(len(need[T.STAGES[2]]) == 36 and all(r['set'] == 'main' and not T30.is_held(r) for r in need[T.STAGES[2]]),
          "pass 2 samples the 36 dev main rows without a tie (controls only)")
    check(all(not (set(map(id, need[a])) & set(map(id, need[b])))
              for a, b in ((T.STAGES[0], T.STAGES[1]), (T.STAGES[0], T.STAGES[2]), (T.STAGES[1], T.STAGES[2]))),
          "no row is sampled twice by the primary juror")
    check((XJ.GO_NET, XJ.GO_RATIO, XJ.GUARD_MIN_NET, XJ.HELD_MIN_NET, XJ.K_JUROR, XJ.TEMPERATURE,
           XJ.PRIMARY_POOL, XJ.JUROR) == (3, 2, -1, 0, 5, 0.8, 'B10', 'deepseek-ai/deepseek-math-7b-rl'),
          "bars and juror frozen: GO net>=+3 W>=2L, guard>=-1, held>=0, k=5, T=0.8, B10, DeepSeek-Math-7B-RL")


def part3():
    print("\nPART 3 - end to end on the stub juror, and the notebooks")
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, 'v31.json')
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            T.run(_args(out=out, max_hours=1e-9))
        check('PARTIAL' in buf.getvalue() and 'SCREEN' not in buf.getvalue(), "a run cut short prints no verdict")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            T.run(_args(out=out, second_juror=True))
        txt = buf.getvalue()
        check('resuming' in txt and 'SCREEN ' in txt and 'GUARD ' in txt and 'HELD ' in txt,
              "the stub run resumes and prints SCREEN, GUARD and HELD")
        check('shared-error floor' in txt and 'Jury-10' in txt and 'second juror' in txt,
              "the summary reports the shared-error floor, the Jury-10 control and the second juror")
        recs = V24._read(out)['rows']
        check(all(len(XJ.votes_of(recs[r['key']]) or []) == 5 for r in ROWS
                  if any(T.needs_jury(r, s) == XJ.JUROR for s in T.STAGES[:3])),
              "every row a pass needs has exactly 5 juror answers")
    for nb in ('MAS_SHT_Kaggle_v31.ipynb', 'MAS_SHT_Colab_v31.ipynb'):
        if not os.path.exists(nb):
            check(False, f"{nb} exists")
            continue
        cells = json.load(open(nb, encoding='utf-8'))['cells']
        ok = True
        for c in cells:
            if c['cell_type'] != 'code':
                continue
            src = ''.join(c['source'])
            src = '\n'.join(('pass  # ' + l) if l.lstrip().startswith('!') else l for l in src.split('\n'))
            try:
                compile(src, nb, 'exec')
            except SyntaxError as e:
                ok = False
                print(f"      {nb}: {e}")
        check(ok, f"every code cell of {nb} compiles")
        check('test_v31.py' in json.dumps(cells) and 'pretest_v31.py' in json.dumps(cells),
              f"{nb} runs test_v31.py before pretest_v31.py")


def main():
    part1()
    part2()
    part3()
    print(f"\n{N[0] - len(FAILS)}/{N[0]} checks passed" + (f"; FAILED: {FAILS}" if FAILS else ''))
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
