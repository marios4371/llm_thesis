"""
[v41.0] v38 replicated on FRESH rows: ADAPT-RS (verifier-gated sampling + Symbolic
Surgery) vs ADAPT-R with the same solver, verifier, policy and code as v38.

WHY
---
v38 is the thesis's strongest result (200 clean GSM-Hard rows: ADAPT-RS 131 vs
ADAPT-R 122, W9 L0, p = 0.004; vs ADAPT W15 L0). Nine wins on one set of 200
rows is a small count. v41 runs the identical system on every remaining clean
GSM-Hard row with numbers >= 1e4 that no version has used, to replicate the
effect on new data and to report it pooled over v38 + v41.

NOTHING NEW IN THE METHOD
-------------------------
This file only swaps v38's manifest and output for new ones and reads the
result; the run itself is pretest_v38.run (Qwen2.5-Math-7B-Instruct, 5
samples at T 0.8, 1536 tokens, Qwen2.5-Math-PRM-7B, the frozen trace_siv audit
and trace_surgery policy, resumes). test_v41.py checks that the frozen files
are v38's.

ROWS (pretest_data/v41_replication.json, seed 41, built once from metadata)
-------------------------------------------------------------------------
GSM-Hard rows that are auditable and clean (gsmhard_audit.py, the same three
defect channels as v38), gold > 0, max number in the text >= 1e4, and in NO
earlier manifest (v38, v36, the v18/v19 rows). Shuffled with seed 41. The run
works through them in order and stops by time (--max-hours), never by outcome.

PRE-REGISTERED (V41_PLAN.md section 3; frozen before any v41 sample exists)
--------------------------------------------------------------------------
Canonical grader |pred - gold| < 1e-3. The analysis set is the complete rows,
in manifest order; a verdict needs at least 150 of them.
  PRIMARY    ADAPT-RS vs ADAPT-R (v38's bars): net >= +3 with W >= 2L ->
             REPLICATED ("significant" if sign p < 0.05); net <= 0 -> NOT
             REPLICATED; otherwise INCONCLUSIVE
  SECONDARY  ADAPT-RS vs SC5 (self-consistency at 5 calls): net >= 0 -> NO
             WORSE THAN THE VOTE
Reported, not decisive: the same comparisons pooled over v38 + v41 (the
v38 files, where present), ADAPT-RS vs ADAPT, the fixed policies, coverage,
resume diagnostics, calls per row.

MODES
-----
    python pretest_v41.py --build-manifest        # once, offline (HF cache); commit it
    python pretest_v41.py --max-hours 8.5         # Kaggle T4 x2; re-run to resume (3 sessions)
    python pretest_v41.py --summary-only [--merge B.json]
    python pretest_v41.py --stub --limit 4        # offline plumbing
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from typing import Dict, List

import pretest_v38 as V38

MANIFEST = 'pretest_data/v41_replication.json'
OUT = 'pretest_v41.json'
OUT_STUB = 'pretest_v41_stub.json'
SEED = 41
USED = (V38.MANIFEST,) + V38.USED_A           # v38's rows, v36's, and v18/v19's
MIN_ROWS = 150

# pre-registered bars (V41_PLAN.md section 3): v38's primary bars, unchanged
PRI_MIN_NET, PRI_RATIO, PRI_ALPHA = V38.PRI_MIN_NET, V38.PRI_RATIO, V38.PRI_ALPHA
V38_RESULTS = ('results_Octomber/pretest_v38.json', 'results_Octomber/pretest_v38_2nd.json',
               'results_Octomber/pretest_v38_Colab.json')


def install() -> None:
    """Point v38's run at v41's rows and files (module globals read at call time)."""
    V38.MANIFEST, V38.OUT, V38.OUT_STUB = MANIFEST, OUT, OUT_STUB


def build_manifest(force: bool = False) -> int:
    if os.path.exists(MANIFEST) and not force:
        print(f"{MANIFEST} exists; refusing to overwrite pre-registered rows (--force to redo)")
        return 1
    from datasets import load_dataset
    import gsmhard_audit as GA
    _, sets = GA.run()
    bad = set(sets['collision']) | set(sets['never_perturbed']) | set(sets['platinum_dropped'])
    used = V38._ids(USED)
    rows = []
    for i, r in enumerate(load_dataset('reasoning-machines/gsm-hard', split='train')):
        pid = f'gsm-hard_{i}'
        g = V38.V36._num(r['target'])
        if (pid in sets['auditable'] and pid not in bad and pid not in used and g is not None and g > 0
                and V38.V36.max_number(r['input']) >= V38.V36.MIN_MAG_A):
            rows.append({'problem_id': pid, 'text': r['input'].strip(), 'gold': g, 'level': 'A',
                         'subject': 'gsm-hard-clean'})
    random.Random(SEED).shuffle(rows)
    with open(MANIFEST, 'w', encoding='utf-8') as fh:
        json.dump({'dataset': 'v41: v38 replicated on every unused clean GSM-Hard row >= 1e4', 'seed': SEED,
                   'n': len(rows), 'used_before': list(USED), 'rows': rows}, fh, ensure_ascii=False, indent=1)
    print(f"wrote {len(rows)} fresh rows -> {MANIFEST}")
    return 0


def verdicts(rows: List[Dict], v38_rows: List[Dict]) -> List[str]:
    R = V38.R
    if len(rows) < MIN_ROWS:
        return [f"none: {len(rows)} complete rows, a verdict needs {MIN_ROWS}"]
    w, l = V38._wl(rows, 'ADAPT-RS', 'ADAPT-R')
    facts = R.wl(rows, 'ADAPT-RS', 'ADAPT-R')
    if w - l >= PRI_MIN_NET and w >= PRI_RATIO * l:
        sig = ' and significant' if R.sign_p(w, l) < PRI_ALPHA else ''
        out = [f"PRIMARY    REPLICATED{sig}: {facts}"]
    elif w - l <= 0:
        out = [f"PRIMARY    NOT REPLICATED: {facts}"]
    else:
        out = [f"PRIMARY    INCONCLUSIVE: {facts}"]
    w, l = V38._wl(rows, 'ADAPT-RS', 'SC5')
    out.append(f"SECONDARY  {'NO WORSE THAN THE VOTE' if w - l >= 0 else 'WORSE THAN THE VOTE'}: "
               f"{R.wl(rows, 'ADAPT-RS', 'SC5')}")
    if v38_rows:
        both = rows + v38_rows
        out.append(f"(reported) v38 + v41 pooled, {len(both)} rows: {R.wl(both, 'ADAPT-RS', 'ADAPT-R')}; "
                   f"{R.wl(both, 'ADAPT-RS', 'SC5')}")
    return out


def summarise(path_list: List[str]) -> int:
    install()
    keep = V38.V21.correct
    V38.V21.correct = V38.canonical
    try:
        rows = V38.load_rows()
        V38._summarise(rows, V38.merged(path_list))
        done = V38._prepare(rows, V38.merged(path_list))
        V38._install_arms(True)
        v38_rows = []
        if any(os.path.exists(f) for f in V38_RESULTS):
            V38.MANIFEST = 'pretest_data/v38_surgery.json'
            v38_rows = [r for r in V38._prepare(V38.load_rows(), V38.merged(list(V38_RESULTS))) if r['pool'] == 'A']
            install()
        print('\n  v41 pre-registered reading (replication of v38):')
        for line in verdicts([r for r in done if r['pool'] == 'A'], v38_rows):
            print('    ' + line)
    finally:
        V38.V21.correct = keep
    return 0


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
    ap.add_argument('--reverse', action='store_true')
    ap.add_argument('--merge', action='append', default=[])
    args = ap.parse_args()
    if args.build_manifest:
        return build_manifest(args.force)
    install()
    if args.summary_only:
        return summarise([args.out or (OUT_STUB if args.stub else OUT)] + args.merge)
    code = V38.run(args)
    if not args.stub:
        summarise([args.out or OUT])
    return code


if __name__ == '__main__':
    sys.exit(main())
