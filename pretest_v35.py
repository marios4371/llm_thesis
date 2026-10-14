"""
[v35.0] Fresh confirmation of Trace-level SIV (trace_siv.py) on MATH-500.

WHY
---
On the 100 stored v34 MATH rows, refuting samples whose own trace holds a
provably false claim lifts v22's ADAPT from 90 to 94 (W4 L0), at the same
calls. Those rows are in-sample for the checker (its false positives were
debugged there), so the claim needs rows nobody has looked at.

ROWS (decided from metadata only, before any sample exists)
-----------------------------------------------------------
pretest_data/v35_math500.json: 80 MATH-500 test problems with a plain numeric
answer, NOT among v34's 100, at LEVELS 3-5, drawn with seed 35. All four rows
where refutation changed the answer in v34 were level 3-5, and levels 1-2
cost half the time for no signal, so the GPU hours go where the effect can be
seen. The scope of the claim is therefore "MATH-500, levels 3-5".

SAMPLES
-------
Exactly v34: 5 samples per row from Qwen2.5-Math-7B-Instruct (v22's sampler and
CoT prompt, T 0.8, 2048 new tokens), every one scored by Qwen2.5-Math-PRM-7B.
The GPU part does not depend on the refutation policy: the audit runs offline
on the saved traces. The policy (trace_siv.refutes) and the checker must be
committed BEFORE the summary is read; the summary prints their hash.

PRE-REGISTERED (V35_PLAN.md section 6; frozen before any sample exists)
----------------------------------------------------------------------
  PRIMARY    ADAPT-R vs ADAPT: net >= +3 with W >= 2L -> SUPPORTED;
             net <= 0 -> REFUTED; otherwise INCONCLUSIVE
  SECONDARY  ADAPT-R vs SC@5: net >= -2 -> NON-INFERIOR
  PRECISION  every FALSE verdict inside a RIGHT sample is printed for a hand check
Reported, not decisive: B5, B5-R, SC5-R, calls/row, the audit statistics.

MODES
-----
    python pretest_v35.py --build-manifest     # once (needs `datasets`); then commit
    python pretest_v35.py --max-hours 8.0      # Kaggle / Colab; re-run to resume
    python pretest_v35.py --summary-only       # re-read, no GPU
    python pretest_v35.py --stub               # offline plumbing
Two machines at once: one runs `--reverse` (rows from the end), the other the
default order; each saves its own file, and
    python pretest_v35.py --summary-only --out A.json --merge B.json
reads them together (a row complete in either file counts once).
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import random
import re
import sys
from typing import Dict, List

import pretest_v21 as V21
import pretest_v24 as V24
import pretest_v34 as V34
import replay_v35 as R
import trace_siv as T

MANIFEST = 'pretest_data/v35_math500.json'
OUT = 'pretest_v35.json'
OUT_STUB = 'pretest_v35_stub.json'
SEED = 35
N_ROWS = 80
LEVELS = ('Level 3', 'Level 4', 'Level 5')

# pre-registered bars (V35_PLAN.md section 6)
PRI_MIN_NET, PRI_RATIO = 3, 2
SEC_MIN_NET = -2


def build_manifest(force: bool = False) -> int:
    if os.path.exists(MANIFEST) and not force:
        print(f"{MANIFEST} exists; refusing to overwrite pre-registered rows (--force to redo)")
        return 1
    from datasets import load_dataset
    ds = load_dataset('HuggingFaceH4/MATH-500', split='test')
    used = {r['problem_id'] for r in V24._read(V34.MANIFEST)['rows']}
    pool = []
    for i, x in enumerate(ds):
        g = V34._to_num(x['answer'].strip())
        if g is None or not re.fullmatch(V34._NUM, x['answer'].strip().replace(' ', '')):
            continue
        pid = f"math500_{i}"
        lv = x['level'] if isinstance(x['level'], str) else f"Level {x['level']}"
        if pid in used or lv not in LEVELS:
            continue
        pool.append({'problem_id': pid, 'text': x['problem'].strip(), 'gold': g, 'level': x['level'],
                     'subject': x['subject'], 'unique_id': x['unique_id']})
    random.Random(SEED).shuffle(pool)
    rows = pool[:N_ROWS]
    with open(MANIFEST, 'w', encoding='utf-8') as fh:
        json.dump({'dataset': 'HuggingFaceH4/MATH-500 (numeric answers, levels 3-5, not in v34)',
                   'seed': SEED, 'n': len(rows), 'pool': len(pool), 'rows': rows},
                  fh, ensure_ascii=False, indent=1)
    print(f"wrote {len(rows)} of {len(pool)} eligible rows -> {MANIFEST}")
    return 0


def _frozen() -> str:
    """Hash of the checker module and of the policy's source, printed with the verdict."""
    src = open(T.__file__, 'rb').read()
    try:
        pol = hashlib.sha256(inspect.getsource(T.refutes).encode()).hexdigest()[:12]
    except (OSError, TypeError):
        pol = 'NOT the committed function (patched at run time)'
    return f"trace_siv.py sha256 {hashlib.sha256(src).hexdigest()[:12]}; refutes() sha256 {pol}"


def verdicts(rows: List[Dict]) -> List[str]:
    out = []
    p = R.wl(rows, 'ADAPT-R', 'ADAPT')
    w = sum(R.right(r, 'ADAPT-R') and not R.right(r, 'ADAPT') for r in rows)
    l = sum(R.right(r, 'ADAPT') and not R.right(r, 'ADAPT-R') for r in rows)
    if w - l >= PRI_MIN_NET and w >= PRI_RATIO * l:
        out.append(f"PRIMARY    SUPPORTED: refutation improves verifier-routed sampling ({p})")
    elif w - l <= 0:
        out.append(f"PRIMARY    REFUTED: {p}")
    else:
        out.append(f"PRIMARY    INCONCLUSIVE: {p}")
    s = R.wl(rows, 'ADAPT-R', 'SC5')
    w = sum(R.right(r, 'ADAPT-R') and not R.right(r, 'SC5') for r in rows)
    l = sum(R.right(r, 'SC5') and not R.right(r, 'ADAPT-R') for r in rows)
    out.append(f"SECONDARY  {'NON-INFERIOR' if w - l >= SEC_MIN_NET else 'NOT MET'} to SC@5: {s}")
    return out


def summarise(rows: List[Dict], recs: Dict) -> int:
    done = []
    for r in rows:
        c = (recs.get(r['key']) or {}).get('C') or []
        if V34.complete({'C': c}):
            s = [dict(x) for x in c[:V34.K]]
            for x in s:
                x['ok'] = V21.correct(x.get('answer'), r['gold'])
                x['audit'] = T.audit(x.get('raw') or '')
            done.append({'set': 'main', 'pid': r['pid'], 'gold': r['gold'], 'level': r['level'], 'C': s})
    print('\n' + '=' * 78)
    print(f"  v35 MATH-500 levels 3-5 (fresh): {len(done)}/{len(rows)} rows complete")
    try:
        T.refutes([], None)
    except NotImplementedError:
        print("  trace_siv.refutes() is not written yet: freeze the policy, commit, then re-read.")
        return 0
    print(f"  frozen: {_frozen()}")
    if done:
        R.table({'MATH-L3-5': done})
        R.audit_stats(done, 'fresh')
        print('\n  FALSE verdicts inside RIGHT samples (hand-check each):')
        n = 0
        for r in done:
            for j, s in enumerate(r['C']):
                if s['ok']:
                    for i, st in enumerate(s['audit']):
                        for t, v in st:
                            if v == T.FALSE:
                                n += 1
                                print(f"    {r['pid']} s{j} step{i}: {t}")
        print(f"    ({n} in total)")
    print('\n  pre-registered reading:')
    if len(done) < len(rows):
        print(f"    none: PARTIAL run, {len(done)}/{len(rows)} rows complete. Re-run the identical command.")
        return 0
    for line in verdicts(done):
        print('    ' + line)
    return 0


def _install(reverse: bool = False) -> None:
    """Point v34's runner at the v35 rows and the v35 reading."""
    V34.MANIFEST, V34.OUT, V34.OUT_STUB = MANIFEST, OUT, OUT_STUB
    V34.summarise = summarise
    if reverse:
        load = V34.load_rows
        V34.load_rows = lambda: list(reversed(load()))


def merged(paths: List[str]) -> Dict:
    """Rows of several result files; a complete record wins over a partial one."""
    out: Dict = {}
    for p in paths:
        if not os.path.exists(p):
            print(f"  (missing: {p})")
            continue
        for k, rec in V24._read(p).get('rows', {}).items():
            if k not in out or (V34.complete(rec) and not V34.complete(out[k])):
                out[k] = rec
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--build-manifest', action='store_true')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--preset', default='qwen_math7b_mixed')
    ap.add_argument('--out', default='')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--stub', action='store_true')
    ap.add_argument('--no-resume', dest='resume', action='store_false')
    ap.add_argument('--max-hours', type=float, default=0.0)
    ap.add_argument('--summary-only', action='store_true')
    ap.add_argument('--reverse', action='store_true', help='work through the rows from the end')
    ap.add_argument('--merge', action='append', default=[], help='more result files (summary only)')
    args = ap.parse_args()
    if args.build_manifest:
        return build_manifest(args.force)
    if args.summary_only:
        _install(False)
        path = args.out or (OUT_STUB if args.stub else OUT)
        rows = V34.load_rows()
        if args.limit:
            rows = rows[:args.limit]
        return summarise(rows, merged([path] + args.merge))
    _install(args.reverse)
    return V34.run(args)


if __name__ == '__main__':
    sys.exit(main())
