"""Replay the WIRED override over the four finished runs.

The point is to exercise the code path solve() now takes -- including
_extract_last_number on the string answer -- rather than the analysis script
that discovered the effect. A wiring bug that silently never fires, or fires
everywhere, would otherwise only show up after a GPU run.
"""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
from scipy.stats import binomtest

from Mas_solver import _extract_last_number
import near_agreement

RUNS = {
    'v14.8': 'results_August/full_mas_20260819.csv',
    'v15.2': 'results_August/mas_full_20260824.csv',
    'v15.5': 'results_August/fullmas_20260830.csv',
    'v15.6': 'results_September/results/MAS_SHT/results/mas_sht_math7b_20260912_094311.csv',
}

def num(x):
    try:
        v = float(x)
        return None if np.isnan(v) else v
    except (TypeError, ValueError):
        return None

def ok(v, g):
    return v is not None and g is not None and (
        abs(v - g) <= 1e-6 * max(abs(g), 1.0) or abs(v - g) < 1e-3)

def main() -> int:
    rows = []
    for run, path in RUNS.items():
        for _, r in pd.read_csv(path).iterrows():
            gold = num(r.gold)
            bp = num(r.siv_blueprint_answer)
            # exactly what solve() does now [v16.4]: the reference is the
            # BASELINE -- an independent derivation -- not the current answer
            cur = _extract_last_number(str(r.baseline_ans))
            fired = near_agreement.structurally_corroborated(cur, bp)
            # `before` must be the pipeline's OWN verdict, not a re-grade:
            # `correct` was computed on the raw answer string and `predicted`
            # has already been through _extract_last_number, so re-grading it
            # disagrees on ~16 rows and would compare against a system that
            # never ran.
            rows.append(dict(run=run, fired=fired,
                             before=bool(r.correct),
                             after=(ok(bp, gold) if fired else bool(r.correct)),
                             base=bool(r.baseline_correct)))
    R = pd.DataFrame(rows)
    fired = R.fired.values
    print(f"rows replayed: {len(R)}   override fired: {fired.sum()}")

    fails = []
    if fired.sum() == 0:
        fails.append("override never fires -- the wiring is dead")
    if fired.sum() > len(R) * 0.15:
        fails.append(f"override fires on {fired.sum()} rows -- far too many, "
                     "the band is not narrow")

    w = int((R.after.values & ~R.before.values).sum())
    l = int((~R.after.values & R.before.values).sum())
    p = binomtest(w, w + l, 0.5).pvalue if w + l else float('nan')
    print(f"\nMAS {R.before.mean():.4f} -> with override {R.after.mean():.4f}  "
          f"({100*(R.after.mean()-R.before.mean()):+.2f}pp)  W={w} L={l}  p={p:.4f}")
    for run, idx in R.groupby('run').groups.items():
        m = R.index.isin(idx)
        print(f"   {run:<7} {R.before[m].mean():.4f} -> {R.after[m].mean():.4f}  "
              f"({100*(R.after[m].mean()-R.before[m].mean()):+.2f}pp)  fires={int(fired[m].sum())}")

    # the discovery run measured +1.33pp / W=10 / L=2 overriding the MAS answer
    if not (9 <= w <= 11 and l <= 3):
        fails.append(f"wired path gives W={w} L={l}; the analysis that found the "
                     f"effect gave W=10 L=2 -- the wiring does not reproduce it")

    # rows where the two already agree EXACTLY must never be touched
    print("\n" + "=" * 68)
    if fails:
        print("FAILURES:")
        for f in fails:
            print("   " + f)
        return 1
    print("REPLAY OK — the wired path reproduces the measured effect")
    return 0

if __name__ == '__main__':
    sys.exit(main())
