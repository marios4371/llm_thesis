"""[v29.0] Guards for the Dispute-to-Question screen. Offline, no GPU.

What these tests exist to prevent is a screen that measures a bug:
  * a comparator that is not v22's rule (it must reproduce B5 78, RA 80,
    RA5 81, B10 81 on the stored pools, and a missing question must change
    nothing);
  * a "factored" answer that can see a solution, or a question that hands
    over one of the disputed values;
  * values mapped to the wrong solution when the pair is reversed;
  * an override on a weak vote, or a knockout that compares a challenger
    with the wrong incumbent;
  * Questioner demos taken from the benchmark;
  * a verdict read off a partial run, or bars that moved after freezing.

Run as `python test_v29.py`.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile

import dispute_question as DQ
import pretest_v21 as V21
import pretest_v24 as V24
import pretest_v29 as T

FAILS = []
N = [0]


def check(cond, label):
    N[0] += 1
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        FAILS.append(label)


def _args(**kw):
    base = dict(v25=T.DEV_V25, out='', preset='qwen_math7b_mixed', stub=True, resume=True,
                max_hours=0.0, summary_only=False)
    base.update(kw)
    return argparse.Namespace(**base)


ROWS = T.load_rows()
MAIN = [r for r in ROWS if r['set'] == 'main']
GUARD = [r for r in ROWS if r['set'] != 'main']


def _perfect(row):
    """A pair getter whose answers always side with the right solution."""
    def gp(a, b):
        ra, rb = sorted((a, b))
        va, vb = row[ra[0]][int(ra[1:])]['answer'], row[rb[0]][int(rb[1:])]['answer']
        oka, okb = V21.correct(va, row['gold']), V21.correct(vb, row['gold'])
        votes = [va] * 5 if oka and not okb else ([vb] * 5 if okb and not oka else [])
        return {'ok': True, 'v1': va, 'v2': vb, 'votes': votes}
    return gp


def part1():
    print("\nPART 1 - the comparator and the disputes")
    check(len(MAIN) == 100 and len(GUARD) == 20, "100 main + 20 guard rows")
    got = {p: sum(V21.correct(DQ.v22_pick(r, p), r['gold']) for r in MAIN) for p in DQ.POOLS}
    check(got == {'B5': 78, 'RA': 80, 'RA5': 81, 'B10': 81},
          f"v22's rule reproduces B5 78, RA 80, RA5 81, B10 81 (got {got})")
    same, incomplete_only_on_disputes = True, True
    for r in ROWS:
        for p in DQ.POOLS:
            pick, info = DQ.resolve(r, p, lambda a, b: None)
            same &= V21.correct(pick, DQ.v22_pick(r, p)) or (pick is None and DQ.v22_pick(r, p) is None)
            incomplete_only_on_disputes &= info['complete'] or info['dispute']
    check(same, "with no question available, DQ keeps v22's pick on every row and pool")
    check(incomplete_only_on_disputes, "a row without a dispute never needs a question")
    ok = True
    for r in ROWS:
        for p in DQ.POOLS:
            cands = DQ.candidates(r, p)
            inc, chs = DQ.contenders(cands)
            if inc is None:
                continue
            ok &= V21.correct(cands[inc]['answer'], DQ.v22_pick(r, p))
            top = max(c['real'] for c in cands if c['real'] is not None)
            ans = [cands[inc]['answer']] + [cands[c]['answer'] for c in chs]
            ok &= len(chs) <= DQ.MAX_CHALLENGERS
            ok &= all(cands[c]['real'] >= top - DQ.EPS for c in chs)
            ok &= all(not V21.correct(x, y) for i, x in enumerate(ans) for y in ans[i + 1:])
    check(ok, "incumbent = v22's pick; challengers: other answers inside the PRM's top tie, distinct, at most 2")
    disp = sum(DQ.contenders(DQ.candidates(r, 'B10'))[1] != [] for r in MAIN)
    check(disp == 64, f"B10 has a dispute on 64 of the 100 main rows (got {disp})")
    ceil = {p: sum(V21.correct(DQ.resolve(r, p, _perfect(r))[0], r['gold']) for r in MAIN) for p in DQ.POOLS}
    check(ceil == {'B5': 81, 'RA': 87, 'RA5': 88, 'B10': 90},
          f"with a perfect answerer: B5 81, RA 87, RA5 88, B10 90 (got {ceil})")


def part2():
    print("\nPART 2 - the Questioner")
    msgs = DQ.question_messages('PROBLEM TEXT', 'first solution', 'x' * 9000)
    check(msgs[0]['role'] == 'system' and len(msgs) == 1 + 2 * len(DQ.Q_DEMOS) + 1,
          "system prompt, two demos, then the pair")
    last = msgs[-1]['content']
    check('PROBLEM TEXT' in last and 'first solution' in last and 'x' * DQ.SOL_CHARS in last
          and 'x' * (DQ.SOL_CHARS + 1) not in last, "the pair is shown in full, each solution cut at SOL_CHARS")
    texts = [r['text'] for r in ROWS]
    check(not any(d[0][:40] in t for d in DQ.Q_DEMOS for t in texts), "the demos are not benchmark problems")
    demos_ok = all(DQ.parse_question(d[3], d[0])['ok'] for d in DQ.Q_DEMOS)
    check(demos_ok, "the demos' own answers parse as usable questions")
    p = DQ.parse_question("QUESTION: How many groups leave early?\nSOLUTION 1: 3\nSOLUTION 2: 2",
                          "These 3 groups of students will need to leave early.")
    check(p['ok'] and p['v1'] == 3.0 and p['v2'] == 2.0, "a well-formed question parses")
    p = DQ.parse_question("QUESTION: What is the cost?\nSOLUTION 1: $1,200\nSOLUTION 2: $960.50", "x")
    check(p['ok'] and p['v1'] == 1200.0 and p['v2'] == 960.5, "money and thousands separators parse")
    for raw, why in (("SOLUTION 1: 3\nSOLUTION 2: 2", 'no question'),
                     ("QUESTION: How many?\nSOLUTION 1: three\nSOLUTION 2: 2", 'no value'),
                     ("QUESTION: How many?\nSOLUTION 1: 4\nSOLUTION 2: 4.0", 'same value')):
        check(not DQ.parse_question(raw, 'The problem.')['ok'], f"unusable: {why}")
    leak = DQ.parse_question("QUESTION: Is the walk 55 minutes long?\nSOLUTION 1: 42\nSOLUTION 2: 55",
                             "It takes them 13 minutes, then another 42 minutes.")
    check(not leak['ok'] and 'disputed value' in leak['reason'],
          "a question that contains a disputed value the problem does not state is refused")
    given = DQ.parse_question("QUESTION: How long after the 42 minutes do they walk back?\n"
                              "SOLUTION 1: 42\nSOLUTION 2: 55",
                              "It takes them 13 minutes, then another 42 minutes.")
    check(given['ok'], "a number the problem itself states may appear in the question")


def part3():
    print("\nPART 3 - the factored answer and the vote")
    leaked = False
    for r in MAIN:
        cands = DQ.candidates(r, 'B10')
        inc, chs = DQ.contenders(cands)
        for c in chs:
            prompt = DQ.answer_prompt(r['text'], 'How many groups leave early?')
            for ref in (cands[inc]['ref'], cands[c]['ref']):
                sol = DQ.raw_of(r, ref).strip()
                lines = [ln for ln in sol.splitlines() if len(ln.strip()) > 30 and ln.strip() not in r['text']]
                leaked |= any(ln.strip() in prompt for ln in lines[:5])
    check(not leaked, "the answer prompt never contains a line of either solution")
    p = DQ.answer_prompt('The problem.', 'The question?')
    check('The problem.' in p and 'The question?' in p and 'Answer:' in p, "it holds the problem and the question")
    check(DQ.overrides([2, 2, 2, 2, 3], 3, 2) and not DQ.overrides([2, 2, 2, 3, 3], 3, 2),
          "strict: 4 of 5 answers for the challenger override, 3 do not")
    check(DQ.overrides([2, 2, 2, 3, 3], 3, 2, 'majority') and not DQ.overrides([2, 2, 3, 3, None], 3, 2, 'majority'),
          "majority (exploratory): more for the challenger than for the incumbent")
    check(DQ.overrides([2.00001, 2, 2, 2, 7], 3, 2), "votes match a value within the repo's tolerance")
    pair = {'ok': True, 'v1': 10.0, 'v2': 20.0}
    check(DQ.values_for(pair, 'C1', 'Q0') == (10.0, 20.0) and DQ.values_for(pair, 'Q0', 'C1') == (20.0, 10.0),
          "values follow the solutions, whichever is the incumbent")
    check(DQ.values_for({'ok': False}, 'C1', 'Q0') is None, "an unusable question decides nothing")


def part4():
    print("\nPART 4 - the knockout")
    sol = lambda a: {'raw': f'... Answer: {a}', 'answer': a, 'prm': [0.9, 1.0]}
    row = {'gold': 30.0, 'C': [sol(10.0), sol(20.0), sol(30.0)] + [sol(10.0)] * 2,
           'Q': [sol(10.0)] * 5}
    asked = []

    def gp(a, b):
        asked.append(DQ.pair_key(a, b))
        ra, rb = sorted((a, b))
        va, vb = row[ra[0]][int(ra[1:])]['answer'], row[rb[0]][int(rb[1:])]['answer']
        # the answers favour the higher value, so 20 beats 10 and then 30 beats 20
        return {'ok': True, 'v1': va, 'v2': vb, 'votes': [max(va, vb)] * 5}
    pick, info = DQ.resolve(row, 'B5', gp)
    check(pick == 30.0 and info['changed'], "two successive overrides reach the third answer")
    check(asked == [DQ.pair_key('C0', 'C1'), DQ.pair_key('C1', 'C2')],
          "the second challenger faces the NEW incumbent (C1), not the original one (C0)")
    check(DQ.pair_key('Q3', 'C0') == DQ.pair_key('C0', 'Q3'), "one question per pair, whatever the order")


def part5():
    print("\nPART 5 - the stub run end to end")
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, 'v29.json')
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            T.run(_args(out=out))
        s = buf.getvalue()
        check('SCREEN' in s and 'GUARD' in s and 'PARTIAL' not in s, "a finished run prints the pre-registered lines")
        check(s.index('pass 1 (B10, strict)') < s.index('pass 2 (all pools)'),
              "the pairs the verdict needs are made first")
        before = json.dumps(V24._read(out)['rows'], sort_keys=True)
        with contextlib.redirect_stdout(io.StringIO()):
            T.run(_args(out=out))
        check(json.dumps(V24._read(out)['rows'], sort_keys=True) == before, "re-running a finished run makes nothing new")
        recs = V24._read(out)['rows']
        keep = {}
        for r in ROWS:
            rec = recs[r['key']]
            need = set()
            DQ.resolve(r, DQ.PRIMARY_POOL, lambda a, b: need.add(DQ.pair_key(a, b)) or rec['pairs'].get(DQ.pair_key(a, b)))
            keep[r['key']] = {'pairs': {k: v for k, v in rec['pairs'].items() if k in need}}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            T.summarise(ROWS, keep)
        check('SCREEN' in buf.getvalue() and 'still partial' in buf.getvalue(),
              "pass 1 alone is enough for the verdict; the other pools are labelled partial")
        cut = os.path.join(d, 'cut.json')
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            T.run(_args(out=cut, max_hours=-1.0))
        check('PARTIAL' in buf.getvalue() and 'SCREEN' not in buf.getvalue(), "a run stopped at once gives no verdict")
        with contextlib.redirect_stdout(io.StringIO()):
            T.run(_args(out=cut))
        check(all(T.complete(r, V24._read(cut)['rows'][r['key']]) for r in ROWS), "the same command resumes it")
    check(T.OUT_STUB != T.OUT, "a stub run never writes the real output file")
    frozen = (DQ.EPS, DQ.PRIMARY_POOL, DQ.MAX_CHALLENGERS, DQ.K_SOLVER, DQ.K_READER, DQ.OVERRIDE_VOTES,
              DQ.ANSWER_TEMPERATURE, DQ.GO_NET, DQ.GO_RATIO, DQ.GUARD_MIN_NET)
    check(frozen == (1e-3, 'B10', 2, 3, 2, 4, 0.8, 3, 2, -1), "the pre-registered bars and settings are the frozen ones")


def part6():
    print("\nPART 6 - the notebooks")
    for name, branch in (('MAS_SHT_Kaggle_v29.ipynb', "'main'"), ('MAS_SHT_Colab_v29.ipynb', "'main'")):
        nb = json.load(open(name, encoding='utf-8'))
        src = '\n'.join(''.join(c['source']) for c in nb['cells'])
        check(f"BRANCH   = {branch}" in src and 'test_v29.py' in src and 'pretest_v29.py' in src
              and '--max-hours' in src and '--summary-only' in src,
              f"{name}: main branch, tests first, the run, the summary")
    nb = json.load(open('MAS_SHT_Colab_v29.ipynb', encoding='utf-8'))
    src = '\n'.join(''.join(c['source']) for c in nb['cells'])
    check("/content/drive/MyDrive/MAS_SHT_v29" in src and "'--out'" in src,
          "Colab: the result file lives on Google Drive")


def main() -> int:
    for p in (part1, part2, part3, part4, part5, part6):
        p()
    print(f"\n{N[0] - len(FAILS)}/{N[0]} checks passed")
    if FAILS:
        print("FAILED:\n  " + "\n  ".join(FAILS))
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
