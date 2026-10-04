"""[v30.0] Guards for the Re-Read Verification screen. Offline, no GPU.

What these tests exist to prevent is a screen that measures a bug:
  * a comparator that is not v22's rule (it must reproduce B5 78, RA 80,
    RA5 81, B10 81 on the stored pools, and a missing score must change
    nothing);
  * a re-read copy that is not the problem verbatim, an answer re-asserted in
    a form the verifier never saw (1.32e+05), or the front repetition (V1)
    built differently from Leviathan et al.'s verbose form;
  * a rule that reaches outside the PRM's contested tie, or that changes a
    pick when the RRV scores carry no information;
  * a verdict read off a partial run, or bars that moved after freezing.

PART 1 needs nothing. PART 2 needs `rrv_key` in reread_verification.py (the
pre-registered rule); until it is written PART 2 fails, and the notebooks
refuse to start the GPU run.

Run as `python test_v30.py`.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import tempfile

import pretest_v21 as V21
import pretest_v24 as V24
import pretest_v30 as T
import reread_verification as RV
import score_prm_v22 as P

FAILS = []
N = [0]


def check(cond, label):
    N[0] += 1
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        FAILS.append(label)


def _args(**kw):
    base = dict(v25=T.DEV_V25, v26=T.DEV_V26, held_traces=T.HELD_TRACES, held_prm=T.HELD_PRM, out='',
                stub=True, resume=True, max_hours=0.0, all_passes=False, summary_only=False)
    base.update(kw)
    return argparse.Namespace(**base)


ROWS = T.load_rows()
MAIN = [r for r in ROWS if r['set'] == 'main']
GUARD = [r for r in ROWS if r['set'] != 'main']
HELD = T.load_held_rows()


def _scores(row, value):
    """Every sample scored in V2 by value(row, ref)."""
    return {ref: {'V2': value(row, ref)} for ref in T.ALL_REFS}


def part1():
    print("\nPART 1 - the comparator, the ties and the verifier's inputs")
    check(len(MAIN) == 100 and len(GUARD) == 20, "100 main + 20 guard rows")
    got = {p: sum(V21.correct(RV.choose(r, p, None, False), r['gold']) for r in MAIN) for p in RV.POOLS}
    check(got == {'B5': 78, 'RA': 80, 'RA5': 81, 'B10': 81},
          f"v22's rule reproduces B5 78, RA 80, RA5 81, B10 81 (got {got})")
    same = all(RV.pick(RV.candidates(r, p), True)[0] == RV.pick(RV.candidates(r, p), False)[0]
               for r in ROWS for p in RV.POOLS)
    check(same, "with no RRV score at all, RRV keeps v22's pick on every row and pool")
    disp = sum(bool(RV.contested_tie(RV.candidates(r, 'B10'))) for r in MAIN)
    check(disp == 64, f"B10 has a contested tie on 64 of the 100 main rows (got {disp})")
    with_right = v22_right = 0
    for r in MAIN:
        cands = RV.candidates(r, 'B10')
        tie = RV.contested_tie(cands)
        if tie and any(V21.correct(cands[i]['answer'], r['gold']) for i in tie):
            with_right += 1
            v22_right += V21.correct(RV.choose(r, 'B10', None, False), r['gold'])
    check((with_right, v22_right) == (57, 48),
          f"57 ties contain the right answer and v22 wins 48 of them (got {with_right}, {v22_right})")
    ok = True
    for r in ROWS:
        cands = RV.candidates(r, 'B10')
        tie = RV.contested_tie(cands)
        if not tie:
            continue
        top = max(c['real'] for c in cands if c['real'] is not None)
        ok &= all(cands[i]['real'] >= top - RV.EPS for i in tie)
        ok &= RV.pick(cands, False)[0] in tie
    check(ok, "a contested tie is inside the PRM's top (1e-3) and contains v22's pick")
    need = [RV.needs(r, ['B10']) for r in ROWS]
    check(all(set(n) <= set(T.ALL_REFS) for n in need) and sum(map(len, need)) > 0,
          f"pass 1 scores only tied B10 samples ({sum(map(len, need))} scores on {sum(1 for n in need if n)} rows)")

    # the inputs
    r = next(x for x in MAIN if x['pid'].endswith('_1268'))
    s = T.sample(r, 'C4')
    user, steps = RV.inputs('V2', r['text'], s['raw'], s['answer'])
    orig = P.split_steps(s['raw'])
    check(user == r['text'] and steps[:-2] == orig, "V2 keeps the problem and the solution's steps untouched")
    check(steps[-2] == RV.REREAD + r['text'].strip(), "V2's re-read step is the problem, verbatim")
    check(steps[-1] == 'So the answer is $\\boxed{16}$.', f"V2 re-asserts the sample's answer ({steps[-1]!r})")
    check(len(steps) == len(orig) + 2, "V2 adds exactly two steps")
    u1, st1 = RV.inputs('V1', r['text'], s['raw'], s['answer'])
    check(u1 == f"{r['text'].strip()}\n{RV.REPEAT}{r['text'].strip()}" and st1 == orig,
          "V1 is Leviathan's verbose repetition of the problem, the solution unchanged")
    u0, st0 = RV.inputs('V0', r['text'], s['raw'], s['answer'])
    check((u0, st0) == (r['text'], orig), "V0 is exactly v22's input")
    up, stp = RV.inputs('V2p', r['text'], s['raw'], s['answer'], prototype='FAMILIAR TEXT')
    check(up == r['text'] and stp[-2] == RV.REREAD + 'FAMILIAR TEXT',
          "V2p changes only the re-read copy (the problem in the user turn stays real)")
    try:
        RV.inputs('V2p', r['text'], s['raw'], s['answer'])
        check(False, "V2p without a prototype is refused")
    except ValueError:
        check(True, "V2p without a prototype is refused")
    fm = {x: RV.fmt_answer(x) for x in (132000.0, 111600.0, 11.11, 0.5, 19.0, -3.0, 1e7)}
    check(fm == {132000.0: '132000', 111600.0: '111600', 11.11: '11.11', 0.5: '0.5', 19.0: '19',
                 -3.0: '-3', 1e7: '10000000'}, f"answers are written plainly, never as exponents ({fm})")
    n_proto = sum(bool(x['prototype']) for x in ROWS)
    check(n_proto == 53, f"53 rows carry a usable, non-identical prototype for V2p (got {n_proto})")
    check('\n' not in RV.REREAD + RV.REPEAT, "the fixed phrases are single-line")

    # the held-out rows
    held = [r for r in HELD if r['set'] == 'held']
    check((len(HELD), len(held)) == (140, 100), f"held-out v21 rows: 140 (100 main) (got {len(HELD)}, {len(held)})")
    check(not ({r['pid'] for r in HELD} & {r['pid'] for r in ROWS}), "no held-out row is a dev row")
    check(all(r['text'] and len(r['C']) == 5 and r['Q'] == [] for r in HELD),
          "every held-out row has its problem text and 5 plain samples, no Reader view")
    hb5 = sum(V21.correct(RV.choose(r, 'B5', None, False), r['gold']) for r in held)
    hdisp = [r for r in held if RV.contested_tie(RV.candidates(r, 'B5'))]
    hwin = [r for r in hdisp if any(V21.correct(c['answer'], r['gold'])
                                    for c in RV.candidates(r, 'B5'))]
    hv22 = sum(V21.correct(RV.choose(r, 'B5', None, False), r['gold']) for r in hdisp
               if any(V21.correct(RV.candidates(r, 'B5')[i]['answer'], r['gold'])
                      for i in RV.contested_tie(RV.candidates(r, 'B5'))))
    check(hb5 == 86 and len(hdisp) == 41 and hv22 == 30,
          f"held-out: v22's B5 is 86, 41 contested ties, v22 wins 30 of the winnable ones "
          f"(got {hb5}, {len(hdisp)}, {hv22}; {len(hwin)} ties hold a right sample anywhere)")
    plans = {s: sum(len(T.plan(r, s)) for r in HELD) for s in T.STAGES}
    check(plans[T.STAGES[1]] > 0 and all(v == 0 for s, v in plans.items() if s != T.STAGES[1]),
          f"held-out rows are scored only in pass 1b ({plans[T.STAGES[1]]} scores)")
    check(all(not T.plan(r, T.STAGES[1]) for r in ROWS), "dev rows are never scored in pass 1b")

    # bars frozen
    check((RV.GO_NET, RV.GO_RATIO, RV.GUARD_MIN_NET, RV.HELD_MIN_NET, RV.EPS, RV.PRIMARY_POOL,
           RV.PRIMARY_FMT) == (3, 2, -1, 0, 1e-3, 'B10', 'V2'),
          "bars frozen: GO net>=+3, W>=2L, guard>=-1, held>=0, EPS 1e-3, B10, V2")


def part2():
    print("\nPART 2 - the rule (needs rrv_key)")
    try:
        RV.rrv_key({'answer': 1.0, 'real': 0.9999, 'rrv': 3.0, 'view': 'C', 'ref': 'C0'}, 0)
    except NotImplementedError:
        check(False, "rrv_key is written (reread_verification.py: the pre-registered rule)")
        return
    # uninformative scores change nothing
    flat = all(V21.correct(RV.choose(r, p, _scores(r, lambda *_: 7.0), True), RV.choose(r, p, None, False))
               or RV.choose(r, p, None, False) is None for r in ROWS for p in RV.POOLS)
    check(flat, "equal RRV scores everywhere -> exactly v22's pick on every row and pool")
    # never outside the tie
    inside = True
    for r in ROWS:
        sc = _scores(r, lambda row, ref: -50.0 if ref in RV.needs(row, ['B10']) else 50.0)
        cands = RV.candidates(r, 'B10', sc)
        i, why = RV.pick(cands, True)
        tie = RV.contested_tie(cands)
        inside &= (i in tie) if tie else (i == RV.pick(cands, False)[0])
    check(inside, "huge scores outside the tie are ignored: the pick stays inside the contested tie")
    # a perfect score gives the ceiling; a perfectly wrong one the floor
    def oracle(sign):
        return lambda row, ref: sign * (10.0 if V21.correct(T.sample(row, ref).get('answer'), row['gold']) else -10.0)
    ceil = {p: sum(V21.correct(RV.choose(r, p, _scores(r, oracle(1)), True), r['gold']) for r in MAIN)
            for p in RV.POOLS}
    floor = sum(V21.correct(RV.choose(r, 'B10', _scores(r, oracle(-1)), True), r['gold']) for r in MAIN)
    check(ceil['B10'] == 90, f"a perfect RRV score gives B10 90, the tie-break ceiling (got {ceil})")
    check(floor == 81 - 48, f"a perfectly wrong RRV score loses all 48 ties v22 wins: B10 {floor}")
    # partial scores inside a tie -> v22
    r = next(x for x in MAIN if x['pid'].endswith('_1268'))
    need = RV.needs(r, ['B10'])
    part = {need[0]: {'V2': 99.0}}
    check(RV.pick(RV.candidates(r, 'B10', part), True)[1] == 'rrv missing',
          "a tie with any sample still unscored keeps v22's pick ('rrv missing')")

    # end to end on the stub verifier
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, 'v30.json')
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            T.run(_args(out=out, max_hours=1e-9))
        txt = buf.getvalue()
        check('PARTIAL run' in txt and 'SCREEN' not in txt, "a run cut short prints no verdict")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            T.run(_args(out=out))
        txt = buf.getvalue()
        check('resuming' in txt and 'SCREEN ' in txt and 'GUARD ' in txt,
              "the stub run resumes and ends with the SCREEN and GUARD lines")
        recs = V24._read(out)['rows']
        check(all(T.primary_complete(r, recs[r['key']]) for r in ROWS),
              "after a full stub run every row is complete for the pre-registered reading")
        check(all(not T.missing(r, recs[r['key']], T.STAGES[1]) for r in HELD),
              "after a full stub run every held-out row is scored")
        check('HELD ' in txt, "the held-out reading is printed")
        check('drift' in recs.get('_meta', {}), "the drift gate ran and its gaps are saved")
        n_v2p = sum(1 for k, v in recs.items() if k != '_meta' for s in v['scores'].values() if 'V2p' in s)
        check(n_v2p > 0, f"the mechanism pass scored V2p samples ({n_v2p})")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            T.summarise(ROWS, recs, True)
        check('AUROC inside' in buf.getvalue() and 'V2 vs V2p' in buf.getvalue(),
              "the summary reports the AUROC and the V2/V2p mechanism line")


def main():
    part1()
    part2()
    print(f"\n{N[0] - len(FAILS)}/{N[0]} checks passed" + (f"; FAILED: {FAILS}" if FAILS else ''))
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
