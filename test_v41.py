"""[v41.0] Guards for the v38 replication on fresh rows (pretest_v41.py). Offline, no GPU.

What these tests exist to prevent:
  * a replication that is not v38: another policy, audit or surgery file, other
    bars, another solver budget;
  * rows that any earlier version already saw (v38, v36, v18/v19, v40), or that
    break v38's row criteria;
  * a run that writes into v38's files (the manifest and output are swapped);
  * a notebook that starts the GPU before this file, or puts a token in a URL.

Run as `python test_v41.py`.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import pretest_v38 as V38
import pretest_v41 as V

FAILS = []
N = [0]


def check(cond, label):
    N[0] += 1
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        FAILS.append(label)


def part1():
    print("\nPART 1 - it is v38")
    import hashlib
    import inspect

    def sha(path):                                   # LF text, as git checks it out on Linux (Kaggle)
        return hashlib.sha256(open(path, 'rb').read().replace(b'\r\n', b'\n')).hexdigest()[:12]
    pol = hashlib.sha256(inspect.getsource(V38.S.surgery_policy).replace('\r\n', '\n').encode()).hexdigest()[:12]
    check(sha(V38.T.__file__) == '483ed23b89b6' and sha(V38.S.__file__) == '42376931d94a' and pol == '8eb4df69f91b',
          "the frozen v35 audit, v38 surgery file and v38 policy")
    check(V38.policy_written(), "the policy is written")
    check((V.PRI_MIN_NET, V.PRI_RATIO, V.PRI_ALPHA, V.MIN_ROWS) == (3, 2, 0.05, 150), "v38's bars; >= 150 rows")
    check(V38.MAX_TOKENS['A'] == 1536 and V38.K == 5 and V38.TEMPERATURE == 0.8, "v38's sampler and budget")
    keep = (V38.MANIFEST, V38.OUT, V38.OUT_STUB)
    V.install()
    check((V38.MANIFEST, V38.OUT) == (V.MANIFEST, V.OUT) and V.OUT != keep[1], "the run writes v41's files, not v38's")
    V38.MANIFEST, V38.OUT, V38.OUT_STUB = keep


def part2():
    print("\nPART 2 - the rows")
    d = json.load(open(V.MANIFEST, encoding='utf-8'))
    rows = d['rows']
    ids = [r['problem_id'] for r in rows]
    check(d['seed'] == 41 and d['n'] == len(rows) == 319 and len(set(ids)) == 319, "seed 41, 319 distinct rows")
    check(all(r['problem_id'].startswith('gsm-hard_') and r['level'] == 'A' and r['gold'] > 0
              and V38.V36.max_number(r['text']) >= 1e4 for r in rows), "clean GSM-Hard, gold > 0, numbers >= 1e4")
    seen = V38._ids(V.USED)
    v40 = {r['pid'] for r in json.load(open('pretest_data/v40_cross_model.json', encoding='utf-8'))['rows']}
    check(not (set(ids) & seen) and not (set(ids) & v40), "no row of v38, v36, v18/v19 or v40")


def part3():
    print("\nPART 3 - the reading")
    have = any(os.path.exists(f) for f in V.V38_RESULTS)
    if not have:
        check(True, "(v38 result files absent: skipped)")
    else:
        keep = V38.V21.correct
        V38.V21.correct = V38.canonical
        try:
            V38._install_arms(True)
            rows = [r for r in V38._prepare(V38.load_rows(), V38.merged(list(V.V38_RESULTS))) if r['pool'] == 'A']
            out = V.verdicts(rows, [])
            check(out[0].startswith('PRIMARY    REPLICATED and significant') and 'W9 L0' in out[0],
                  "on v38's own rows the reading says REPLICATED and significant (W9 L0)")
            check(V.verdicts(rows[:100], [])[0].startswith('none'), "fewer than 150 rows: no verdict")
        finally:
            V38.V21.correct = keep
    d = tempfile.mkdtemp()
    out = os.path.join(d, 's.json')
    r = subprocess.run([sys.executable, 'pretest_v41.py', '--stub', '--limit', '3', '--out', out],
                       capture_output=True, text=True, encoding='utf-8', errors='replace')
    recs = json.load(open(out, encoding='utf-8'))['rows'] if os.path.exists(out) else {}
    check(r.returncode == 0 and len(recs) == 3 and all(V38.complete(x) for x in recs.values())
          and set(recs) <= {x['problem_id'] for x in json.load(open(V.MANIFEST, encoding='utf-8'))['rows']},
          "the stub run fills 3 v41 rows")


def part4():
    print("\nPART 4 - the notebook")
    nb = 'MAS_SHT_Kaggle_v41.ipynb'
    if not os.path.exists(nb):
        check(False, f"{nb} exists")
        return
    cells = json.load(open(nb, encoding='utf-8'))['cells']
    code = '\n'.join(''.join(c['source']) for c in cells if c['cell_type'] == 'code')
    ok = True
    for c in cells:
        if c['cell_type'] == 'code':
            src = '\n'.join(('pass  # ' + l) if l.lstrip().startswith('!') else l
                            for l in ''.join(c['source']).split('\n'))
            try:
                compile(src, nb, 'exec')
            except SyntaxError as e:
                ok = False
                print(f"      {e}")
    check(ok, f"every code cell of {nb} compiles")
    t, r = code.find("'test_v41.py'"), code.find('pretest_v41.py --max-hours')
    check(t >= 0 and r > t, f"{nb} runs test_v41.py before the run")
    check('_tok}@' not in code and '_token}@' not in code and '@github.com' not in code, f"{nb} puts no token in a URL")


def main():
    for p in (part1, part2, part3, part4):
        p()
    print(f"\n{N[0] - len(FAILS)}/{N[0]} checks passed" + (f"; FAILED: {FAILS}" if FAILS else ''))
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
