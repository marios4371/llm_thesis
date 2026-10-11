"""[v33.0] Guards for the Verifier-Gated Reading Arbitration confirmation. Offline, no GPU.

What these tests exist to prevent is a confirmation that measures a bug:
  * comparators that are not the stored ones (dev: B5 78, SC@5 68, B3 73, RA 80);
  * a gate that peeks at samples it is not allowed to see (C4, C5, the
    reader view) or that drifted from v22's 0.85;
  * VGRA charged the wrong number of calls (3 ungated, 6 gated);
  * a verdict read off a partial run, a notebook cell that does not compile,
    or bars that moved after freezing.

Run as `python test_v33.py`.
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import subprocess
import sys
import tempfile

import gated_reading as G
import pretest_v30 as T30
import pretest_v33 as T

FAILS = []
N = [0]


def check(cond, label):
    N[0] += 1
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        FAILS.append(label)


ROWS = T30.load_rows()
MAIN = [r for r in ROWS if r['set'] == 'main']
GUARD = [r for r in ROWS if r['set'] == 'guard']


def part1():
    print("\nPART 1 - the dev numbers (in-sample; they motivated v33, they are not its claim)")
    check((len(MAIN), len(GUARD)) == (100, 20), "100 dev main + 20 guard rows")
    got = {a: sum(G.right(r, a) for r in MAIN) for a in ('B5', 'S5', 'B3', 'S3', 'RA')}
    check(got == {'B5': 78, 'S5': 68, 'B3': 73, 'S3': 66, 'RA': 80},
          f"the stored comparators are reproduced (got {got})")
    v = sum(G.right(r, 'VGRA') for r in MAIN)
    check(v == 80 and abs(G.calls(MAIN, 'VGRA') - 3.48) < 1e-9 and sum(G.gated(r) for r in MAIN) == 16,
          f"VGRA on dev: 80/100 at 3.48 calls/row, 16 rows gated (got {v}, {G.calls(MAIN, 'VGRA'):.2f})")
    check(sum(G.right(r, 'ADAPT5') for r in MAIN) == 76,
          "the same gate spending plain samples instead of the Reader (ADAPT5): 76")
    check(all(G.complete(r) for r in ROWS), "every dev row has C1..C5 and Q1..Q2 scored")


def part2():
    print("\nPART 2 - the rule")
    blind = True
    for r in ROWS:
        x = copy.deepcopy(r)
        for s in x['C'][3:] + x['Q']:
            s['prm'] = [0.0]
            s['answer'] = -999.0
        blind &= G.gated(x) == G.gated(r)
    check(blind, "the gate reads C1..C3 only (C4, C5 and the reader view cannot move it)")
    check(all(G.vgra(r) == (G.ARMS['B3'](r)[0], 3) for r in ROWS if not G.gated(r)),
          "an ungated row is exactly B3, charged 3 calls")
    inside = all(G.vgra(r)[1] == 6 and any(G.V21.correct(G.vgra(r)[0], c['answer'])
                                           for c in G.cands(r, 3, 2) if c['answer'] is not None)
                 for r in ROWS if G.gated(r))
    check(inside, "a gated row answers from {C1..C3, Q1, Q2} and is charged 6 calls")
    r = next(x for x in MAIN if G.gated(x))
    y = copy.deepcopy(r)
    for s in y['C'][:3]:
        s['prm'] = [0.99, 0.999]
    check(not G.gated(y), "three confident plain samples (min step >= 0.85) do not call the Reader")
    check((G.GATE_K, G.GATE_MIN, G.READ_K, G.SUP_MIN_NET, G.SUP_RATIO, G.SUP_ALPHA, G.NI_MIN_NET,
           G.NI_MAX_CALLS, G.BEAT_MIN_NET, G.GUARD_MIN_NET) == (3, 0.85, 2, 5, 2, 0.05, 0, 4.0, 3, -1),
          "rule and bars frozen: gate 3 samples / min step 0.85 / 2 read samples; "
          "SC@5 net>=+5 W>=2L p<0.05; B5 net>=0 (beat: >=+3) at <=4 calls; guard >=-1")


def part3():
    print("\nPART 3 - plumbing, the verdicts and the notebooks")
    cmd = T.generation_cmd('x.json', 8.0, False)
    check(cmd[2:] == ['pretest_v25.py', '--out', 'x.json', '--max-hours', '8.0'],
          "the draws are pretest_v25.py's confirmation mode, unchanged, resumable via --max-hours")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        T.summarise(copy.deepcopy(ROWS), 'DEV')
    txt = buf.getvalue()
    check('PRIMARY 1  SUPPORTED' in txt and 'PRIMARY 2  NON-INFERIOR' in txt and 'GUARD' in txt,
          "on the dev rows the verdict code prints SUPPORTED / NON-INFERIOR / GUARD")
    part = copy.deepcopy(ROWS)
    part[0]['C'][4]['prm'] = None
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        T.summarise(part, 'DEV')
    check('PARTIAL' in buf.getvalue() and 'PRIMARY' not in buf.getvalue().split('pre-registered reading')[1],
          "a partial run prints no verdict")
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, 'v33.json')
        r = subprocess.run([sys.executable, 'pretest_v33.py', '--stub', '--out', out],
                           capture_output=True, text=True)
        check(r.returncode == 0 and 'CONFIRMATION' in r.stdout and 'PRIMARY 1' in r.stdout,
              "end to end through pretest_v25.py --stub: draws, scores, prints the verdict")
    for nb in ('MAS_SHT_Kaggle_v33.ipynb', 'MAS_SHT_Colab_v33.ipynb'):
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
        t = js.find("'test_v33.py'")
        run = [i for i in (js.find('pretest_v33.py --max-hours'), js.find("'pretest_v33.py', '--out'")) if i >= 0]
        check(t >= 0 and "'test_v25.py'" in js and bool(run) and t < min(run),
              f"{nb} runs test_v25.py and test_v33.py before the GPU run")


def main():
    part1()
    part2()
    part3()
    print(f"\n{N[0] - len(FAILS)}/{N[0]} checks passed" + (f"; FAILED: {FAILS}" if FAILS else ''))
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
