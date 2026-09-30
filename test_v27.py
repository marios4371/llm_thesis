"""[v27.0] Guards for Null-Hypothesis Verification. Offline, no models.

What these tests exist to prevent is a screen that measures something other
than the null test:
  * a rule that can pick a candidate the verifier rates clearly worse, or that
    changes v22's choice when the null test carries no information;
  * null rewards computed against anything but the prototype, or real rewards
    that move;
  * pools that do not reproduce the stored v25 numbers;
  * prototypes written with a different prompt than v26's;
  * a verdict read off a partial run, or bars that moved after freezing.

Run as `python test_v27.py`.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import tempfile

import null_hypothesis as NH
import pretest_v21 as V21
import pretest_v24 as V24
import pretest_v27 as T
import prototype_contrast as PC

FAILS = []
N = [0]


def check(cond, label):
    N[0] += 1
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        FAILS.append(label)


def c(ans, real, null, view='C'):
    return {'view': view, 'answer': ans, 'real': real, 'null': null}


def part1():
    print("\nPART 1 - the rule")
    tie = [c(24, 1.0, 0.99), c(6, 0.9995, 0.20), c(3, 0.80, 0.01)]
    i, why = NH.pick(tie, True)
    check(i == 1 and why == 'null test', "in a contested tie the candidate the null rejects wins")
    check(NH.pick(tie, False)[0] == 0, "without the null test the rule is v22's (best real reward)")
    check(NH.pick(tie, True)[0] != 2, "a candidate outside the tie is never chosen, however low its null")
    same_null = [c(24, 1.0, 0.5), c(6, 0.9995, 0.5)]
    check(NH.pick(same_null, True)[0] == NH.pick(same_null, False)[0],
          "equal null rewards keep v22's choice: an uninformative null changes nothing")
    one = [c(24, 1.0, 0.9), c(24, 0.9999, 0.1), c(6, 0.5, 0.0)]
    check(NH.pick(one, True) == (0, 'no contested tie'), "a tie with a single answer is left to v22")
    miss = [c(24, 1.0, None), c(6, 1.0, 0.1)]
    check(NH.pick(miss, True)[1] == 'null reward missing' and NH.pick(miss, True)[0] == 0,
          "a missing null reward in the tie falls back to v22")
    check(NH.pick([c(None, 1.0, 0.1)], True)[0] is None, "no answer, no pick")
    rec = {'gold': 6.0, 'prototype': {'ok': True, 'identical': True},
           'C': [{'answer': 24, 'prm': [1.0], 'pprm': [0.9]}, {'answer': 6, 'prm': [0.9995], 'pprm': [0.1]}]
           + [{'answer': 24, 'prm': [0.5], 'pprm': [0.5]}] * 3,
           'Q': [{'answer': 24, 'prm': [0.1], 'pprm': [0.1]}] * 5}
    check(NH.choose(rec, 'B5', True) == NH.choose(rec, 'B5', False) == 24,
          "an identical prototype makes NHV exactly v22")
    rec['prototype'] = {'ok': True, 'identical': False}
    check(NH.choose(rec, 'B5', True) == 6 and NH.right(rec, 'B5', True), "an edited prototype enables the test")
    rec['prototype'] = {'ok': False, 'identical': True}
    check(NH.choose(rec, 'B5', True) == 24, "a failed prototype disables the test")
    check(NH.POOLS == {'B5': (5, 0), 'RA': (2, 2), 'RA5': (3, 2), 'B10': (5, 5)} and NH.EPS == 1e-3,
          "pools and the tie width are as documented")


def part2():
    print("\nPART 2 - the stored pools")
    rows = T.load_rows()
    main = [r for r in rows if r['set'] == 'main']
    guard = [r for r in rows if r['set'] == 'guard']
    check(len(main) == 100 and len(guard) == 20, "100 main + 20 guard rows from the v25 dev file")
    got = {p: sum(NH.right(r, p, False) for r in main) for p in NH.POOLS}
    check(got == {'B5': 78, 'RA': 80, 'RA5': 81, 'B10': 81},
          f"v22's rule reproduces the stored numbers {got}")
    reused = [r for r in rows if r.get('prototype', {}).get('source') == 'v26']
    check(len(reused) == 78 and all(r['prototype'].get('ok') for r in reused),
          "78 prototypes are reused from the v26 pilot")
    check(sum('prototype' not in r for r in rows) == 42, "42 rows still need a prototype")
    v26 = {r['pid']: r['prototype']['text'] for r in V24._read(T.DEV_V26)['rows'] if 'prototype' in r}
    check(all(r['prototype']['text'] == v26[r['pid']] for r in reused), "reused prototypes are v26's text, unchanged")
    check(all(len(r['C']) == 5 and len(r['Q']) == 5 and all(s['prm'] for v in 'CQ' for s in r[v])
              for r in rows), "every row has 5 + 5 samples with stored real rewards")


class _Rec:
    def __init__(self):
        self.seen = []

    def score(self, problem, steps):
        self.seen.append(problem)
        return [0.5] * len(steps)


def _args(**kw):
    base = dict(v25=T.DEV_V25, v26=T.DEV_V26, out='', preset='qwen_math7b_mixed', limit=8, stub=True,
                resume=True, max_hours=0.0, summary_only=False)
    base.update(kw)
    return argparse.Namespace(**base)


def part3():
    print("\nPART 3 - the runner")
    seen_msgs = []

    class _Reader(T._StubReader):
        def call_model(self, msgs, **kw):
            seen_msgs.append(msgs)
            return super().call_model(msgs, **kw)
    rec = _Rec()
    orig_r, orig_p = T.build_reader, T.build_prm
    T.build_reader = lambda args: _Reader()
    T.build_prm = lambda args, rows: rec
    try:
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, 'o.json')
            with contextlib.redirect_stdout(io.StringIO()):
                T.run(_args(out=out))
            data = V24._read(out)
            rows = {r['pid']: r for r in T.load_rows()}
            check(seen_msgs and all(m == PC.proto_messages(m[-1]['content'].split('Problem: ', 1)[-1])
                                    for m in seen_msgs),
                  "new prototypes are written with v26's exact prompt")
            protos = {r['pid']: r['prototype']['text'] for r in data['rows']}
            usable = [r for r in data['rows'] if NH.usable_null(r)]
            check(set(rec.seen) <= {protos[r['pid']] for r in usable} and len(rec.seen) == 10 * len(usable),
                  "every C and Q sample of a usable row is scored, and only ever against its prototype")
            check(all(all(s.get('pprm') is None for v in 'CQ' for s in r[v])
                      for r in data['rows'] if not NH.usable_null(r)),
                  "rows with an identical or failed prototype get no null rewards")
            check(all(s['prm'] == rows[r['pid']][v][i]['prm'] for r in data['rows'] for v in 'CQ'
                      for i, s in enumerate(r[v])), "the real rewards are untouched")
            check(all('raw' not in s for r in data['rows'] for v in 'CQ' for s in r[v])
                  and all('text' not in r for r in data['rows']), "the result file holds no raw texts")
            n = len(rec.seen)
            with contextlib.redirect_stdout(io.StringIO()):
                T.run(_args(out=out))
            check(len(rec.seen) == n, "re-running resumes: nothing is scored twice")
            out2 = os.path.join(d, 'cut.json')
            with contextlib.redirect_stdout(io.StringIO()):
                T.run(_args(out=out2, limit=4))
                T.run(_args(out=out2, limit=8, max_hours=-1.0))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                T.summarise(V24._read(out2)['rows'], stub=False)
            check('PARTIAL run' in buf.getvalue() and 'SCREEN' not in buf.getvalue(),
                  "a run stopped early gives no verdict")
    finally:
        T.build_reader, T.build_prm = orig_r, orig_p


def _row(pid, gold, v22_right, nh_right, tpl=0, tag='main'):
    """A row whose pools give the requested outcomes for RA (C1,C2,Q1,Q2)."""
    right_ans, wrong_ans = gold, gold + 1
    a0 = right_ans if v22_right else wrong_ans
    a1 = right_ans if nh_right else (wrong_ans if not v22_right else wrong_ans)
    C = [{'answer': a0, 'prm': [1.0], 'pprm': [0.9]}, {'answer': a1 if a1 != a0 else a0, 'prm': [0.9999],
                                                      'pprm': [0.1 if nh_right != v22_right else 0.95]}]
    C += [{'answer': wrong_ans, 'prm': [0.1], 'pprm': [0.1]}] * 3
    Q = [{'answer': wrong_ans, 'prm': [0.1], 'pprm': [0.1]}] * 5
    return {'pid': pid, 'set': tag, 'gold': gold, 'template': tpl, 'C': C, 'Q': Q,
            'prototype': {'ok': True, 'identical': False}}


def part4():
    print("\nPART 4 - verdicts and bars")
    base = [_row(f'b{i}', 5.0, True, True, tpl=i) for i in range(10)]
    go = base + [_row(f'w{i}', 5.0, False, True, tpl=20 + i) for i in range(4)]
    check(sum(NH.right(r, 'RA', True) for r in go) == 14 and sum(NH.right(r, 'RA', False) for r in go) == 10,
          "the synthetic rows do what they say")
    check(NH.screen_verdict(go, [])[0].startswith('SCREEN   GO'), "net +4 with no losses -> GO")
    weak = base + [_row('w1', 5.0, False, True), _row('w2', 5.0, False, True), _row('l1', 5.0, True, False)]
    check(NH.screen_verdict(weak, [])[0].startswith('SCREEN   WEAK'), "net +1 -> WEAK")
    stop = base + [_row('l1', 5.0, True, False)]
    check(NH.screen_verdict(stop, [])[0].startswith('SCREEN   STOP'), "net -1 -> STOP")
    g = [_row('g1', 5.0, True, False, tag='guard'), _row('g2', 5.0, True, False, tag='guard')]
    check(NH.screen_verdict(go, g)[1].endswith('HARM'), "guard net -2 -> HARM")
    check((NH.GO_NET, NH.GO_RATIO, NH.GUARD_MIN_NET, NH.PRIMARY_POOL) == (3, 2, -1, 'RA'),
          "the bars are the ones frozen on 2026-09-30")
    doc = T.__doc__
    check('frozen 2026-09-30' in doc and 'net >= +3 rows with W >= 2L' in doc and 'NH-RA - RA >= -1' in doc,
          "the docstring states the same bars")


def main() -> int:
    for p in (part1, part2, part3, part4):
        p()
    print(f"\n{N[0] - len(FAILS)}/{N[0]} checks passed")
    if FAILS:
        print("FAILED:\n  " + "\n  ".join(FAILS))
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
