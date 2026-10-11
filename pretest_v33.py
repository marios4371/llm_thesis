"""
[v33.0] Fresh-row confirmation of Verifier-Gated Reading Arbitration (VGRA).

The samples are drawn by pretest_v25.py's confirmation mode, UNCHANGED (56
offline checks): 100 GSM-Symbolic P2 rows + 20 GSM-Plus distractor rows
(seed 48, disjoint from every earlier version's rows), each with 5 plain
samples, one Reader call, 2 samples under the reading, all scored by v22's
PRM against the original text. Every arm below is read off those samples;
VGRA uses the Reader and its samples only on the rows its gate selects, and
is charged for them only there.

The rule and the dev evidence are in gated_reading.py; the plan in V33_PLAN.md.

PRE-REGISTERED (V33_PLAN.md section 5; frozen 2026-10-11 before any
confirmation sample exists)
-------------------------------------------------------------------
  PRIMARY 1 (100 main rows)  VGRA vs SC@5: net >= +5 with W >= 2L and sign
                             p < 0.05 -> SUPPORTED; net <= 0 -> REFUTED
  PRIMARY 2 (100 main rows)  VGRA vs B5 (v22): net >= +3 with W >= 2L at
                             <= 4.0 calls/row -> BEATS; net >= 0 at <= 4.0
                             calls/row -> NON-INFERIOR; otherwise NOT MET
  GUARD    (20 rows)         VGRA - B5 >= -1
Reported, not decisive: B3, S3, RA (always-on reading, v25), ADAPT5 (the same
gate spending plain samples instead of the Reader: does the READER matter, or
just more samples?), accuracy on the gated rows, pooled dev + confirmation.

MODES
-----
    python pretest_v33.py --max-hours 8.0      # Kaggle T4 x2: draws (resumable), then the reading
    python pretest_v33.py --summary-only       # re-read pretest_v25.json, no GPU
    python pretest_v33.py --dev                # the in-sample dev numbers (stored v22/v25 rows)
    python pretest_v33.py --stub               # offline plumbing through pretest_v25 --stub
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Dict, List

import gated_reading as G
import pretest_v24 as V24
import pretest_v30 as T30

CONFIRM_OUT = 'pretest_v25.json'      # pretest_v25.py's own confirmation file
STUB_OUT = 'pretest_v33_stub.json'
ORDER = ('VGRA', 'B5', 'S5', 'B3', 'S3', 'RA', 'ADAPT5')


def scrub() -> None:
    """Kaggle saves the clone as Output; strip a GitHub token from its git
    metadata first (see pretest_v32.scrub_git_credentials)."""
    try:
        import pretest_v32 as T32
        T32.scrub_git_credentials(os.path.dirname(os.path.abspath(__file__)))
    except Exception as exc:
        print(f"  (credential scrub skipped: {type(exc).__name__}: {exc})")


def generation_cmd(out: str, max_hours: float, stub: bool) -> List[str]:
    cmd = [sys.executable, '-u', 'pretest_v25.py', '--out', out]
    if stub:
        cmd += ['--stub', '--limit', '8']
    if max_hours:
        cmd += ['--max-hours', str(max_hours)]
    return cmd


def table(rows: List[Dict], title: str) -> None:
    if not rows:
        return
    print('\n' + '=' * 78)
    print(f"  {title}: {len(rows)} rows, {len({r.get('template') for r in rows})} templates")
    print(f"    {'arm':7s} {'right':>6s} {'calls/row':>10s}")
    for arm in ORDER:
        print(f"    {arm:7s} {sum(G.right(r, arm) for r in rows):6d} {G.calls(rows, arm):10.2f}")
    for x, y in (('VGRA', 'S5'), ('VGRA', 'B5'), ('VGRA', 'RA'), ('VGRA', 'ADAPT5'), ('B5', 'S5')):
        c = G.paired(rows, x, y)
        print(f"    {x:6s} vs {y:6s}: W={c['w']:2d} L={c['l']:2d} net={c['net']:+3d}  sign p={c['p']:.4f}")
    print('    ' + G.gated_rows_line(rows))


def summarise(recs: List[Dict], label: str) -> int:
    done = [r for r in recs if G.complete(r)]
    main = [r for r in done if r['set'] == 'main']
    guard = [r for r in done if r['set'] == 'guard']
    table(main, f'{label} MAIN (P2)')
    table(guard, f'{label} GUARD (GSM-Plus)')
    print('\n  pre-registered reading:')
    if len(done) < len(recs):
        print(f"    none: PARTIAL run, {len(done)}/{len(recs)} rows complete. Re-run the identical command.")
        return 0
    if not main:
        print("    none: no complete main rows")
        return 0
    for line in G.verdicts(main, guard):
        print('    ' + line)
    return 0


def dev_rows() -> List[Dict]:
    return T30.load_rows()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='')
    ap.add_argument('--max-hours', type=float, default=0.0)
    ap.add_argument('--summary-only', action='store_true')
    ap.add_argument('--dev', action='store_true')
    ap.add_argument('--stub', action='store_true')
    args = ap.parse_args()
    if args.dev:
        print("IN-SAMPLE: these dev rows motivated the composition; never quote them as the claim.")
        return summarise(dev_rows(), 'DEV (v22 seed 45, stored samples)')
    out = args.out or (STUB_OUT if args.stub else CONFIRM_OUT)
    if not args.summary_only:
        if not args.stub:
            scrub()
        r = subprocess.run(generation_cmd(out, args.max_hours, args.stub))
        if r.returncode:
            print(f"pretest_v25.py exited with {r.returncode}")
            return r.returncode
    if not os.path.exists(out):
        print(f"missing {out}")
        return 2
    return summarise(V24._read(out)['rows'], 'CONFIRMATION (v25 seed 48, fresh)')


if __name__ == '__main__':
    sys.exit(main())
