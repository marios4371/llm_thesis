"""[v34.0] Guards for the MATH-500 generalization run. Offline, no GPU.

What these tests exist to prevent is a run that measures a bug:
  * a parser that misreads MATH answers (\\frac, \\boxed with text, an
    'Answer:' past 4000 characters);
  * a manifest that moved, or rows that are not numeric-answer MATH-500;
  * ADAPT5 that is not v22's frozen rule (gate 3 samples, 0.85), or that is
    charged the wrong number of calls;
  * a verdict read off a partial run, a notebook cell that does not compile,
    or bars that moved after freezing.

Run as `python test_v34.py`.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile

import gated_reading as G
import pretest_v34 as T

FAILS = []
N = [0]


def check(cond, label):
    N[0] += 1
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        FAILS.append(label)


def part1():
    print("\nPART 1 - the answer parser")
    cases = [
        (r"so the answer is $\boxed{\frac{3}{4}}$.", 0.75),
        (r"so $\boxed{\dfrac{-5}{2}}$", -2.5),
        ("Answer: 12\n\nthen we see \\boxed{1,234}", 1234.0),
        (r"\boxed{x=3} first, then the final \boxed{-7}", -7.0),
        (r"The angle is $\boxed{90^\circ}$", 90.0),
        (r"\boxed{25\%}", 25.0),
        (r"\boxed{\text{(A)}}", None),
        ("The answer is 42.", 42.0),
        ("x" * 5000 + "\n\nAnswer: 17", 17.0),
    ]
    for raw, want in cases:
        got = T.parse_answer(raw)
        check(got == want, f"parse {raw[-40:]!r} -> {want} (got {got})")


def part2():
    print("\nPART 2 - the manifest, the rule, the bars")
    d = json.load(open(T.MANIFEST, encoding='utf-8'))
    rows = T.load_rows()
    check(d['seed'] == 34 and d['n'] == 100 and len(rows) == 100 and len({r['pid'] for r in rows}) == 100,
          "100 distinct MATH-500 rows, seed 34")
    check(all(isinstance(r['gold'], float) for r in rows) and {r['level'] for r in rows} == {1, 2, 3, 4, 5},
          "every row has a numeric gold; all five levels are present")
    check(rows[0]['pid'] == d['rows'][0]['problem_id'] and d['pool'] == 317,
          "the manifest is the frozen one (pool of 317 numeric-answer problems)")
    check((G.GATE_K, G.GATE_MIN) == (3, 0.85) and T.ARMS[0] == 'ADAPT5',
          "ADAPT5 is v22's frozen rule: gate on the best of 3, min step 0.85")
    mk = lambda prm3: {'C': [{'answer': 1.0, 'prm': p} for p in prm3] +
                            [{'answer': 2.0, 'prm': [1.0]}, {'answer': 2.0, 'prm': [1.0]}], 'gold': 1.0}
    sure, doubt = mk([[0.9, 0.99]] * 3), mk([[0.5, 0.99]] * 3)
    check(G.ARMS['ADAPT5'](sure)[1] == 3 and G.ARMS['ADAPT5'](doubt)[1] == 5,
          "3 calls when the verifier is sure, 5 when it doubts")
    check((T.K, T.TEMPERATURE, T.MAX_TOKENS, T.SUP_MIN_NET, T.SUP_RATIO, T.SUP_ALPHA, T.NI_MIN_NET,
           T.NI_MAX_CALLS, T.BEAT_MIN_NET) == (5, 0.8, 2048, 5, 2, 0.05, 0, 4.0, 3),
          "sampler and bars frozen: k=5, T=0.8, 2048 tokens; SC@5 net>=+5 W>=2L p<0.05; "
          "B5 net>=0 (beat >=+3) at <=4 calls")


def part3():
    print("\nPART 3 - plumbing and the notebooks")
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, 'v34.json')
        r = subprocess.run([sys.executable, 'pretest_v34.py', '--stub', '--limit', '6', '--out', out,
                            '--max-hours', '1e-9'], capture_output=True, text=True)
        check(r.returncode == 0 and 'PARTIAL' in r.stdout and 'PRIMARY' not in r.stdout,
              "a run cut short prints no verdict")
        r = subprocess.run([sys.executable, 'pretest_v34.py', '--stub', '--limit', '6', '--out', out],
                           capture_output=True, text=True)
        check(r.returncode == 0 and 'resuming' in r.stdout and 'PRIMARY 1' in r.stdout and 'by level' in r.stdout,
              "the stub run resumes, prints the verdict and the per-level table")
        recs = json.load(open(out, encoding='utf-8'))['rows']
        check(len(recs) == 6 and all(len(v['C']) == 5 and all(s.get('prm') for s in v['C']) for v in recs.values()),
              "every row gets 5 scored samples")
    for nb in ('MAS_SHT_Kaggle_v34.ipynb', 'MAS_SHT_Colab_v34.ipynb'):
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
        t = js.find("'test_v34.py'")
        run = [i for i in (js.find('pretest_v34.py --max-hours'), js.find("'pretest_v34.py', '--out'")) if i >= 0]
        check(t >= 0 and bool(run) and t < min(run), f"{nb} runs test_v34.py before the GPU run")
        if 'Kaggle' in nb:
            src = ''.join(''.join(c['source']) for c in cells)
            check('{_gh_token}@' not in src, f"{nb} never puts the GitHub token in a URL")


def main():
    part1()
    part2()
    part3()
    print(f"\n{N[0] - len(FAILS)}/{N[0]} checks passed" + (f"; FAILED: {FAILS}" if FAILS else ''))
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
