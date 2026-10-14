"""[v35.0] Guards for Trace-level SIV (trace_siv.py). Offline, no GPU.

What these tests exist to prevent:
  * a FALSE verdict on a claim that is true or out of scope. Every case in
    PART 1's "never FALSE" list is a false positive the audit once produced
    on the stored pools (`11 remainder 8` read as 88, sympy's E, `f(0)`, a
    case split, a unit conversion, a run-on chain, integer division);
  * a checker that stops seeing the slips the method rests on
    (`64^2 = 64 (mod 11)`, `9999 = 3*3*11*11*11`, the math500_214 anchor,
    the correlated P2 sum `61 + 8 + 13 + 10 + 58 = 140`);
  * anything but numbers, names and operators reaching SymPy's parser;
  * a refutation policy that breaks the obvious invariants;
  * an ADAPT-R that is not v22's ADAPT when nothing is refuted.

Run as `python test_v35.py`.
"""
from __future__ import annotations

import os
import sys

import gated_reading as G
import trace_siv as T

FAILS = []
N = [0]


def check(cond, label):
    N[0] += 1
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        FAILS.append(label)


def _verdict(step: str) -> str:
    vs = {v for _, v in T.check_step(step)}
    return 'F' if T.FALSE in vs else ('T' if vs else '-')


def part1():
    print("\nPART 1 - the claim checker")
    must_false = [
        r"\(8^4 = (8^2)^2 = 64^2 \equiv 64 \pmod{11}.\)",                     # math500_415
        r"\[9999 = 3 \times 3 \times 11 \times 11 \times 11 = 3^2 \times 11^3\]",  # math500_272
        r"\[ 61 + 8 + 13 + 10 + 58 = 140 \]",                                  # p2_185, 4 of 5 samples
        r"\[ 4.32 + 4.95 + 64.80 = 73.07 \]",                                  # p2_1535, 3 of 5 samples
        r"\[ 19 \text{ minutes} + 18 \text{ minutes} = 36 \text{ minutes} \]",
        r"So 12 + 15 = 28 cookies.",
        r"\[ 9 \times 1111 = 3^2 \times 11 \times 91 \]",
    ]
    for s in must_false:
        check(_verdict(s) == 'F', f"FALSE: {s[:60]!r}")
    must_true = [
        r"\(4 \times 23 \times \left(1 - \frac{1}{4}\right) + 3 \times 23 = 69 + 69 = 138\)",
        r"\[ 19 \text{ minutes} + 18 \text{ minutes} = 37 \text{ minutes} \]",   # p2_721's twist step
        r"\[ \frac{448}{5} = 89.6 \]",
        r"\[ \frac{100}{3} \approx 33.33 \]",
        r"\[ 1,400 - 768 = 632 \]",
        r"\[\binom{5}{2} = 10\]",
        r"\[ 3.75 \times 0.80 = ¥3.00 \]",
        r"\[ 60 \times \frac{1}{30} = 2 \text{ miles} \]",
    ]
    for s in must_true:
        check(_verdict(s) == 'T', f"TRUE:  {s[:60]!r}")
    never_false = [
        r"\[ 129 \div 11 = 11 \text{ remainder } 8 \]",
        r"\[ 1 \div 7 = 0 \text{ remainder } 1 \]",
        r"\[ 629 \div 7 = 89 \]",
        r"\[ 60 \text{ minutes} = \frac{60}{60} \text{ hours} \]",
        r"\[ 1.25 \text{ hours} = 1.25 \times 60 \]",
        r"\[ 48 \text{ donuts} = 4 \text{ dozen} \]",
        r"\[ E = 36 + 11 \]",
        r"\[ x = 0 \quad \text{or} \quad x - 5 = 0 \]",
        r"\[ f(0) = \frac{1}{2}(0)^4 - 2(0)^2 + \frac{9}{2} \]",
        r"\[ \frac{361}{400} = 0.9025 \times 100 = 90.25 \]",
        r"\[ 50 - 1.25 - 1 = 47.75 + 1 = 48.75 \]",
        r"\[ \frac{78 \text{ miles}}{13 \text{ miles per 30 minutes}} = 6 \times 30 \text{ minutes} \]",
        r"\[ \text{Sakura's current age} + 2 = 3 \times 15 \]",
        r"\[ 2x + 3 = 11 \]",
        r"\[ 0.\overline{1331} = \frac{1331}{9999} \]",
        r"\[ 2\frac{1}{2} = 2.5 \]",
    ]
    for s in never_false:
        check(_verdict(s) != 'F', f"never FALSE: {s[:60]!r}")
    a = T.check_anchors([r"\[ 345 = 180 + |P| + 2|P| - 30 - 75 - 75 + 15 \]",
                         r"\[ 345 = 180 + 3|P| - 165 + 15 \]"])
    check([v for _, _, v in a] == [T.FALSE], "anchor: math500_214's dropped -15 is FALSE")
    a = T.check_anchors([r"\[ x = 0 \quad \text{or} \quad x - 5 = 0 \]"])
    check(not a, "anchor: a case split on 0 is not checked")


def part2():
    print("\nPART 2 - only numbers, names and operators reach SymPy")
    for s in ("__import__('os').system('echo x')", r"(1).__class__", r"\text{a}.real",
              "lambda: 1", "[1 for x in ()]", "1 if 1 else 2"):
        check(T.normalize(s) is None, f"normalize refuses {s!r}")


def part3():
    print("\nPART 3 - the refutation policy (yours) keeps the invariants")
    none = [[], [('2 + 2 = 4', T.TRUE)], []]
    try:
        r0 = T.refutes(none, [0.99, 0.2, 0.99])
        r1 = T.refutes([[('61 + 8 = 70', T.FALSE)]], [0.10])
        r2 = T.refutes([], None)
    except NotImplementedError:
        check(False, "trace_siv.refutes is implemented (V35_PLAN.md section 3)")
        return
    check(r0 is False, "no FALSE claim -> never refuted, however low the PRM step")
    check(r1 is True, "a FALSE claim in a step the PRM also doubts (0.10) -> refuted")
    check(r2 is False, "an empty audit and no PRM scores -> not refuted")


def part4():
    print("\nPART 4 - ADAPT-R is v22's ADAPT when nothing is refuted")
    keep = T.refutes
    T.refutes = lambda v, p: False
    try:
        rows = []
        for path in ('results_September/preset_v22.json',):
            if os.path.exists(path):
                import json
                for r in json.load(open(path, encoding='utf-8'))['rows']:
                    rows.append({'C': r['samples'][:5], 'gold': r['gold']})
        if not rows:
            check(True, "(stored pool absent in this checkout: skipped)")
            return
        same = all(T.adapt_r(r['C']) == G.ARMS['ADAPT5'](r) for r in rows)
        check(same, f"ADAPT-R == gated_reading.ARMS['ADAPT5'] on all {len(rows)} v22 rows (answer and calls)")
    finally:
        T.refutes = keep
    s = [{'answer': 1.0, 'prm': [0.99, 0.99]}, {'answer': 2.0, 'prm': [0.5, 1.0]}, {'answer': 3.0, 'prm': [0.9]}]
    T_keep = T.refutes
    T.refutes = lambda v, p: True
    try:
        check(T.adapt_r(s + s[:2]) == (2.0, 5), "everything refuted -> v22's pick over all five, 5 calls")
    finally:
        T.refutes = T_keep


def part5():
    print("\nPART 5 - the fresh rows, the bars, the notebooks")
    import json
    import pretest_v35 as V
    d = json.load(open(V.MANIFEST, encoding='utf-8'))
    v34 = {r['problem_id'] for r in json.load(open('pretest_data/v34_math500.json', encoding='utf-8'))['rows']}
    ids = [r['problem_id'] for r in d['rows']]
    check(d['seed'] == 35 and d['n'] == 80 and len(set(ids)) == 80, "manifest: seed 35, 80 distinct rows")
    check(not (set(ids) & v34), "manifest: no row from v34 (fresh)")
    check(all(r['level'] in (3, 4, 5, 'Level 3', 'Level 4', 'Level 5') for r in d['rows']), "manifest: levels 3-5 only")
    check((V.PRI_MIN_NET, V.PRI_RATIO, V.SEC_MIN_NET) == (3, 2, -2), "bars frozen: PRIMARY +3 with W >= 2L, SECONDARY >= -2")
    check((T.GATE_K, T.FULL_K, T.GATE_MIN) == (3, 5, 0.85), "ADAPT-R uses v22's gate (3, 5, 0.85)")
    for nb, run in (('MAS_SHT_Colab_v35.ipynb', "'pretest_v35.py', '--out'"),
                    ('MAS_SHT_Kaggle_v35.ipynb', 'pretest_v35.py --reverse --max-hours')):
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
        js = json.dumps(cells)
        t, r = js.find("test_v35.py"), js.find(run.replace("'", "\\'") if False else run)
        check(t >= 0 and r >= 0 and t < r, f"{nb} runs test_v35.py before the GPU run")
        if 'Kaggle' in nb:
            check('{_gh_token}@' not in ''.join(''.join(c['source']) for c in cells),
                  f"{nb} never puts the GitHub token in a URL")


def main():
    part1()
    part2()
    part3()
    part4()
    part5()
    print(f"\n{N[0] - len(FAILS)}/{N[0]} checks passed" + (f"; FAILED: {FAILS}" if FAILS else ''))
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
