"""[v32.0] Guards for the Reading Jury screen. Offline, no GPU.

What these tests exist to prevent is a screen that measures a bug:
  * a comparator that is not v22's rule (B10 81 / B5 78 on dev, B5 86 held-out);
  * a rule that counts SAMPLES or VIEWS instead of families (the same-family
    Reader would then vote twice: B10 74, the measured failure), or that
    reaches outside the PRM's contested tie;
  * foreign-view samples drawn from a failed reading being counted as a
    foreign witness;
  * a verdict read off a partial run, a notebook cell that does not compile
    (the v30 Kaggle run died on one), or bars that moved after freezing.

PART 2 checks the rule you write in reading_jury.jury_choice; it fails until
the rule exists. Run as `python test_v32.py`.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import sys
import tempfile

import null_hypothesis as NH
import pretest_v21 as V21
import pretest_v24 as V24
import pretest_v30 as T30
import pretest_v32 as T
import reading_jury as RJ
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
                held=True, preset='qwen_math7b_mixed', temperature=RJ.TEMPERATURE,
                max_tokens=RJ.MAX_TOKENS, out='', stub=True, resume=True, max_hours=0.0,
                summary_only=False)
    base.update(kw)
    return argparse.Namespace(**base)


ROWS = T.load_rows(_args())
DEV = [r for r in ROWS if not T30.is_held(r)]
MAIN = [r for r in DEV if r['set'] == 'main']
GUARD = [r for r in DEV if r['set'] == 'guard']
HELD = [r for r in ROWS if r['set'] == 'held']


def _acc(rows, pool, use_rule, rule=None):
    return sum(RJ.right(r, pool, use_rule, rule) for r in rows)


def _g(*fams, real=1.0):
    """A synthetic tie group with these families."""
    f = frozenset(fams)
    return {'families': f, 'tie_families': f, 'n_pool': len(f), 'n_tie': len(f), 'real': real,
            'answer': 1.0, 'rep': 0}


def _keep(groups):            # plumbing rule: always v22
    return None


def _last(groups):            # plumbing rule: always the last group (stress: stays inside the tie)
    return len(groups) - 1


def _plumbing(groups):        # plumbing rule for the stub run only, NOT the method
    s = [len(g['families']) for g in groups]
    return s.index(max(s)) if s.count(max(s)) == 1 and max(s) >= 2 else None


def part1():
    print("\nPART 1 - the comparator, the pools and the families")
    check((len(MAIN), len(GUARD), len(HELD)) == (100, 20, 100), "100 dev main + 20 guard + 100 held-out main rows")
    got = {p: _acc(MAIN, p, False) for p in ('B5', 'B10')}
    check(got == {'B5': 78, 'B10': 81}, f"v22 reproduced on dev through reading_jury (got {got})")
    check(_acc(HELD, 'B5', False) == 86, "v22 reproduced on the held-out rows: B5 86")
    b9 = _acc(MAIN, 'B9', False)
    check(76 <= b9 <= 81, f"B9 (C5 + Q4, v22's rule) is computed: {b9}")
    same = all(RJ.choose(r, 'B10', False) == r_ans for r in DEV
               for r_ans in [RV.candidates(r, 'B10')[NH.pick(RV.candidates(r, 'B10'), False)[0]]['answer']])
    check(same, "reading_jury's candidates give exactly reread_verification's v22 pick on every dev row")
    fam_ok = all(c['family'] == 'qwen' for r in DEV for c in RJ.candidates(r, 'B10'))
    check(fam_ok, "every stored C and Q sample is the Solver's own family ('qwen')")
    check(RJ.family_of('F1', {'family': 'qwen'}) == 'qwen' and RJ.family_of('F2', {'family': 'granite'}) == 'granite'
          and RJ.family_of('F1', {}) == 'phi', "a failed-reading sample counts as Qwen's; a usable one as its reader's")
    first = True
    for r in DEV:
        c = RJ.candidates(r, 'B10')
        tie = RV.contested_tie(c)
        if tie:
            gs = RJ.tie_groups(c, tie)
            first &= gs[0]['rep'] == NH.pick(c, False)[0] and len(gs) >= 2
    check(first, "tie_groups puts v22's pick first on every contested row")
    keep = all(RJ.choose(r, 'B10', True, _keep) == RJ.choose(r, 'B10', False) for r in DEV)
    check(keep, "a rule that returns None is exactly v22")
    inside = True
    for r in DEV:
        c = RJ.candidates(r, 'B10')
        tie = RV.contested_tie(c)
        i, _ = RJ.rj_pick(c, _last)
        inside &= (i in tie) if tie else (i == NH.pick(c, False)[0])
    check(inside, "whatever the rule returns, the pick stays inside the contested tie")
    r = next(x for x in MAIN if x['pid'].endswith('_1268'))
    c = RJ.candidates(r, 'B10')
    try:
        RJ.rj_pick(c, lambda g: 99)
        raised = False
    except ValueError:
        raised = True
    check(raised, "an out-of-range choice raises instead of silently picking something")
    check(RJ.POOLS['RJ'] == (('C', 5), ('F1', 2), ('F2', 2)) and RJ.POOLS['B10'] == (('C', 5), ('Q', 5)),
          "RJ = C5 + F1x2 + F2x2 (9 solver samples) against B10 = C5 + Q5 (10)")
    check((RJ.GO_NET, RJ.GO_RATIO, RJ.GUARD_MIN_NET, RJ.HELD_MIN_NET, RJ.K_READ, RJ.TEMPERATURE,
           RJ.PRIMARY_POOL, RJ.COMPARATOR_POOL, RJ.READER_MODEL['F1'], RJ.READER_MODEL['F2'])
          == (3, 2, -1, 0, 2, 0.8, 'RJ', 'B10', 'microsoft/Phi-3.5-mini-instruct',
              'ibm-granite/granite-3.1-8b-instruct'),
          "bars and agents frozen: GO net>=+3 W>=2L, guard>=-1, held>=0, k=2, T=0.8, Phi-3.5-mini + Granite-3.1-8B")


def part2():
    print("\nPART 2 - the rule you wrote (reading_jury.jury_choice)")
    try:
        RJ.jury_choice([_g('qwen'), _g('qwen')])
    except NotImplementedError:
        check(False, "jury_choice is written (it still raises NotImplementedError)")
        return
    got = {p: _acc(MAIN, p, True) for p in ('B5', 'B10')}
    check(got == {'B5': 78, 'B10': 81},
          f"requirement 1: with only Qwen's family the rule is v22 (B5 78, B10 81; got {got})")
    check(all(RJ.choose(r, 'B10', True) == RJ.choose(r, 'B10', False) for r in DEV)
          and all(RJ.choose(r, 'B5', True) == RJ.choose(r, 'B5', False) for r in HELD),
          "requirement 1, row by row: no stored row changes its pick")
    for gs in ([_g('qwen'), _g('phi', 'granite')], [_g('qwen'), _g('qwen', 'phi', 'granite')],
               [_g('qwen'), _g('qwen'), _g('phi', 'granite')]):
        want = len(gs) - 1
        check(RJ.jury_choice(gs) == want,
              f"requirement 2: {[sorted(g['families']) for g in gs]} -> group {want}")
    rng = random.Random(32)
    fams = ['qwen', 'phi', 'granite']
    valid = True
    for _ in range(3000):
        gs = []
        for _ in range(rng.randint(2, 4)):
            f = frozenset(x for x in fams if rng.random() < 0.5) or frozenset({'qwen'})
            t = frozenset(x for x in f if rng.random() < 0.7) or frozenset([sorted(f)[0]])
            gs.append({'families': f, 'tie_families': t, 'n_pool': rng.randint(len(f), 9),
                       'n_tie': rng.randint(len(t), 6), 'real': 1 - rng.random() * 1e-3,
                       'answer': rng.random(), 'rep': 0})
        k = RJ.jury_choice(gs)
        valid &= k is None or (isinstance(k, int) and not isinstance(k, bool) and 0 <= k < len(gs))
    check(valid, "requirement 3: on 3000 random ties it returns None or a valid index")
    print("      what your rule does in the cases the requirements leave open:")
    for gs in ([_g('qwen'), _g('phi')], [_g('qwen', 'phi'), _g('granite')],
               [_g('qwen', 'phi'), _g('qwen', 'granite')], [_g('qwen'), _g('phi'), _g('granite')],
               [_g('phi'), _g('qwen', 'granite')]):
        print(f"        {[sorted(g['families']) for g in gs]} -> {RJ.jury_choice(gs)}")


def part3():
    print("\nPART 3 - end to end on stub agents, and the notebooks")
    check(T.split_batched(['a0', 'a1', 'b0', 'b1'], 2, 2) == [['a0', 'a1'], ['b0', 'b1']],
          "batched sampling: prompt 0's samples first, then prompt 1's")
    try:
        T.split_batched(['a', 'b', 'c'], 2, 2)
        bad = False
    except ValueError:
        bad = True
    check(bad, "a batched call that returns the wrong number of sequences raises")
    saved = RJ.jury_choice
    RJ.jury_choice = _plumbing          # the stub run must exercise the verdicts whatever the user wrote
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, 'v32.json')
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                T.run(_args(out=out, max_hours=1e-9))
            check('PARTIAL' in buf.getvalue() and 'SCREEN' not in buf.getvalue(),
                  "a run cut short prints no verdict")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                T.run(_args(out=out))
            txt = buf.getvalue()
            check('resuming' in txt and 'SCREEN ' in txt and 'GUARD ' in txt and 'HELD ' in txt,
                  "the stub run resumes and prints SCREEN, GUARD and HELD")
            check('shared-error floor' in txt and 'per-sample accuracy' in txt and 'readings usable' in txt,
                  "the summary reports the mechanism: accuracy per view, readings, shared-error floor")
            recs = V24._read(out)['rows']
            full = all(len(recs[r['key']].get(v) or []) == RJ.K_READ
                       and all(s.get('prm') is not None and s.get('family') for s in recs[r['key']][v])
                       and v in recs[r['key']]['readings']
                       for r in ROWS for v in RJ.READER_VIEWS)
            check(full, "every row has both readings and 2 scored, family-labelled samples per foreign view")
            fallback = [(recs[r['key']]['readings']['F2'], recs[r['key']]['F2']) for r in ROWS]
            lab = all((s['family'] == 'granite') == bool(rd['ok']) for rd, ss in fallback for s in ss)
            check(lab and any(not rd['ok'] for rd, _ in fallback),
                  "samples drawn after a failed reading are labelled 'qwen', the rest 'granite'")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rows = T.load_rows(_args())
                T.summarise(rows, V24._read(out)['rows'])
            check('SCREEN ' in buf.getvalue(), "--summary-only reads the file back to the same verdict")
    finally:
        RJ.jury_choice = saved
    for nb in ('MAS_SHT_Kaggle_v32.ipynb', 'MAS_SHT_Colab_v32.ipynb'):
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
        js = json.dumps(cells)
        t = js.find("'test_v32.py'")            # quoted: 'pretest_v32.py' contains 'test_v32.py'
        runs = [i for i in (js.find('pretest_v32.py --held'), js.find("'pretest_v32.py', '--out'")) if i >= 0]
        check(t >= 0 and bool(runs) and t < min(runs) and '--held' in js,
              f"{nb} runs test_v32.py before the pretest_v32.py --held run")


def main():
    part1()
    part2()
    part3()
    print(f"\n{N[0] - len(FAILS)}/{N[0]} checks passed" + (f"; FAILED: {FAILS}" if FAILS else ''))
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
