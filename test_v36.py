"""[v36.0] Guards for the large arithmetic-heavy run of Trace-level SIV. Offline, no GPU.

What these tests exist to prevent:
  * the two v35 false-positive classes coming back (`11111111_2`, a claim the
    step itself calls false) or the fixes eating real slips;
  * a manifest that moved, overlaps a pool, mixes in a defective GSM-Hard row,
    or is not interleaved (a partial run would then cover only one pool);
  * bars that moved after freezing, a token budget that is not 1536;
  * a notebook that starts the GPU before test_v36.py, or runs the wrong direction.

Run as `python test_v36.py`.
"""
from __future__ import annotations

import json
import sys

import pretest_v36 as V
import trace_siv as T

FAILS = []
N = [0]


def check(cond, label):
    N[0] += 1
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        FAILS.append(label)


def _v(step: str) -> str:
    vs = {v for _, v in T.check_step(step)}
    return 'F' if T.FALSE in vs else ('T' if vs else '-')


def part1():
    print("\nPART 1 - v35.1 checker fixes")
    check(_v(r"\[11111111_2 = 1 \cdot 2^7 + 1 \cdot 2^6 + 1 \cdot 2^0\]") != 'F', "base subscript 11111111_2 is not refuted")
    check(_v(r"For \( p = 2 \) this gives \( 0 \equiv 1 \pmod{2} \), which is false.") != 'F',
          "a claim the step calls false is not refuted")
    check(_v(r"Wait, \( 77 + 13 + 702 = 892 \) is a mistake.") != 'F', "a self-corrected claim is not refuted")
    for s in (r"\[ 61 + 8 + 13 + 10 + 58 = 140 \]", r"\[ 2009 = 7 \times 49 \]",
              r"\[ 4.32 + 4.95 + 64.80 = 73.07 \]", r"\[ 3473626 \times 12 = 41683502 \]",
              r"\(8^4 = 64^2 \equiv 64 \pmod{11}.\)"):
        check(_v(s) == 'F', f"real slip still refuted: {s[:50]!r}")
    check(_v(r"\[ 3473626 \times 12 = 41683512 \]") == 'T', "large-number product checked exactly")
    check(_v(r"\[ 5,798,892 - 1,200,000 = 4,598,892 \]") == 'T', "thousands separators in large numbers")


def part2():
    print("\nPART 2 - the manifest")
    d = json.load(open(V.MANIFEST, encoding='utf-8'))
    rows = d['rows']
    ids = [r['problem_id'] for r in rows]
    check(d['seed'] == 36 and d['n'] == 200 and len(set(ids)) == 200, "seed 36, 200 distinct rows")
    by = {k: [r for r in rows if r['level'] == k] for k in 'ABC'}
    check([len(by[k]) for k in 'ABC'] == [100, 50, 50], "pools A/B/C = 100/50/50")
    check([r['level'] for r in rows[:8]] == list('ABACABAC'), "interleaved A,B,A,C from the start")
    check([r['level'] for r in rows[-4:]] == list('ABAC'), "interleaved to the end (the reverse half mixes too)")
    check(all(r['problem_id'].startswith('gsm-hard_') and r['gold'] > 0 and V.max_number(r['text']) >= V.MIN_MAG_A
              for r in by['A']), "A: GSM-Hard, gold > 0, max number >= 1e4")
    check(all(V.max_number(r['text']) >= V.MIN_MAG_B for r in by['B']), "B: max number >= 1e3")
    check(all(r['subject'].endswith('digit expansion') for r in by['B'])
          and all(r['subject'].endswith('integer-decimal-fraction conversion') for r in by['C']),
          "B/C perturbation types")
    used = set()
    for f in V.USED_BEFORE:
        used |= {r['problem_id'] for r in json.load(open(f, encoding='utf-8'))['rows']}
    check(not (set(ids) & used), "no row from the v18/v19 pretests")
    try:
        audit = json.load(open('gsmhard_audit_rows.json', encoding='utf-8'))['rows']
        bad = set(audit['collision']) | set(audit['never_perturbed']) | set(audit['platinum_dropped'])
        check(not (set(r['problem_id'] for r in by['A']) & bad) and
              set(r['problem_id'] for r in by['A']) <= set(audit['auditable']),
              "A: every row auditable and clean in the GSM-Hard audit")
    except FileNotFoundError:
        check(True, "(gsmhard_audit_rows.json absent in this checkout: skipped)")


def part3():
    print("\nPART 3 - bars, budget, the rule")
    check((V.PRI_MIN_NET, V.PRI_RATIO, V.PRI_ALPHA, V.SEC_MIN_NET) == (5, 2, 0.05, 0), "bars frozen")
    check(V.MAX_TOKENS == 1536, "1536 new tokens")
    check((T.GATE_K, T.FULL_K, T.GATE_MIN) == (3, 5, 0.85), "v22's gate")
    check(T.refutes([[('1 + 1 = 3', T.FALSE)]], None) and not T.refutes([[('1 + 1 = 2', T.TRUE)]], None),
          "policy: any FALSE claim refutes")


def part4():
    print("\nPART 4 - the notebooks")
    for nb, run in (('MAS_SHT_Kaggle_v36.ipynb', 'pretest_v36.py --max-hours 8.5'),
                    ('MAS_SHT_Colab_v36.ipynb', "'pretest_v36.py', '--reverse'")):
        cells = json.load(open(nb, encoding='utf-8'))['cells']
        ok = True
        for c in cells:
            if c['cell_type'] != 'code':
                continue
            src = '\n'.join(('pass  # ' + l) if l.lstrip().startswith('!') else l
                            for l in ''.join(c['source']).split('\n'))
            try:
                compile(src, nb, 'exec')
            except SyntaxError as e:
                ok = False
                print(f"      {nb}: {e}")
        check(ok, f"every code cell of {nb} compiles")
        code = '\n'.join(''.join(c['source']) for c in cells if c['cell_type'] == 'code')
        t, r = code.find("'test_v36.py'"), code.find(run)
        check(t >= 0 and r >= 0 and t < r, f"{nb} runs test_v36.py before `{run}`")
        check('v35' not in code and 'v34' not in code, f"{nb} code cells mention no older version")
        if 'Kaggle' in nb:
            check('{_gh_token}@' not in code, f"{nb} never puts the GitHub token in a URL")


def main():
    part1()
    part2()
    part3()
    part4()
    print(f"\n{N[0] - len(FAILS)}/{N[0]} checks passed" + (f"; FAILED: {FAILS}" if FAILS else ''))
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
