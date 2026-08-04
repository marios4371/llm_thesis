"""
Offline counterfactual replay of the selection layer.

WHY
---
MAS-SHT's accuracy is decided twice: once by the agents that produce candidate
answers, and once by the layer that picks between them. Only the first costs GPU
hours. This script re-runs the SECOND on a finished CSV, so "what if the
do-no-harm invariant anchored on the Programmer instead of the zero-shot
baseline?" costs seconds instead of a 12h commit.

It needs the per-candidate columns added in v14.1 (`cand_primary`,
`cand_baseline`, `cand_blueprint_eval`). Runs from before that have no record of
what the Programmer actually answered -- which is exactly why the anchor
question went unanswered for so long -- and this script will say so rather than
guess.

THE QUESTION IT EXISTS TO SETTLE
--------------------------------
On the v12.0 problem set (n=150, identical problems) PAL-style code scored
88.00% and zero-shot CoT 80.67%, while MAS-SHT finished at 82.67% -- losing 13
problems PAL got right and winning only 5. That points at the anchor: v13.0
hardcoded do-no-harm to the weaker of the two derivations.

The counter-evidence is that on the v12.2 mixed run the primary scored 8/55
versus the baseline's 29/55 -- but only over judge_fallback rows, a subset
selected BY primary-vs-baseline disagreement, which says nothing about either
one's marginal accuracy. This script computes the marginal numbers.

USAGE
    python replay_selection.py <run.csv> [more.csv ...]
"""

import sys
import numpy as np
import pandas as pd


def to_num(x):
    try:
        return float(str(x).replace(",", "").strip())
    except Exception:
        return np.nan


def eq(a, b, rel=1e-6):
    """Numeric match with a relative tolerance.

    NOT the grader used in the runs (that one is absolute 1e-3, which mis-marks
    gsm-hard's million-scale answers). Reported side by side below so the
    difference is visible rather than silently baked in.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    with np.errstate(invalid="ignore"):
        return (~np.isnan(a)) & (~np.isnan(b)) & (
            np.abs(a - b) <= rel * np.maximum(np.abs(b), 1.0))


def replay(path):
    d = pd.read_csv(path)
    print("=" * 74)
    print(f"{path}   n={len(d)}   solver_version={d.solver_version.dropna().unique()}")
    print("=" * 74)

    cand_cols = [c for c in d.columns if c.startswith("cand_")]
    if not cand_cols:
        print("  NO cand_* columns -> pre-v14.1 run. The candidates each problem")
        print("  actually produced were never written out, so no selection policy")
        print("  can be replayed from it. Re-run with v14.1+ to get these columns.")
        return

    gold = d.gold.map(to_num).values
    print(f"  candidate columns: {cand_cols}\n")

    # ---- marginal accuracy of every candidate, and of the shipped answer ----
    print("  MARGINAL ACCURACY (rel_tol=1e-6, over rows where the candidate exists)")
    rows = []
    for c in ["predicted", "baseline_ans"] + cand_cols:
        if c not in d.columns:
            continue
        v = d[c].map(to_num).values
        have = ~np.isnan(v)
        ok = eq(v, gold)
        rows.append((c, int(have.sum()), int(ok.sum()),
                     100.0 * ok.sum() / max(1, have.sum()),
                     100.0 * ok.sum() / len(d)))
    w = max(len(r[0]) for r in rows)
    print(f"    {'candidate'.ljust(w)}  present  correct   acc|present   acc|all")
    for name, present, ok, acc_p, acc_a in rows:
        print(f"    {name.ljust(w)}  {present:7d}  {ok:7d}   {acc_p:9.2f}%  {acc_a:8.2f}%")

    # ---- the anchor counterfactual -----------------------------------------
    if "cand_primary" in d.columns and "cand_baseline" in d.columns:
        prim = d.cand_primary.map(to_num).values
        base = d.cand_baseline.map(to_num).values
        pred = d.predicted.map(to_num).values
        p_ok, b_ok, pred_ok = eq(prim, gold), eq(base, gold), eq(pred, gold)

        print("\n  PRIMARY vs BASELINE, paired over all rows")
        print(f"    primary right, baseline wrong : {int((p_ok & ~b_ok).sum())}")
        print(f"    baseline right, primary wrong : {int((b_ok & ~p_ok).sum())}")
        print(f"    both right                    : {int((p_ok & b_ok).sum())}")
        print(f"    both wrong                    : {int((~p_ok & ~b_ok).sum())}")
        try:
            from scipy.stats import binomtest
            a, b = int((p_ok & ~b_ok).sum()), int((b_ok & ~p_ok).sum())
            if a + b:
                print(f"    McNemar exact p = {binomtest(min(a, b), a + b, 0.5).pvalue:.4f}")
        except Exception:
            pass

        # Rows the invariant actually decides: triage ended in a *_default.
        if "sht_triage" in d.columns:
            anch = d.sht_triage.astype(str).str.endswith("_default").values
            n_anch = int(anch.sum())
            print(f"\n  ROWS DECIDED BY THE ANCHOR ({n_anch}/{len(d)} "
                  f"= {100.0*n_anch/max(1,len(d)):.1f}%)")
            if n_anch:
                print(f"    anchored on baseline -> correct: {int((b_ok & anch).sum())}/{n_anch}"
                      f" = {100.0*(b_ok & anch).sum()/n_anch:.1f}%")
                print(f"    anchored on primary  -> correct: {int((p_ok & anch).sum())}/{n_anch}"
                      f" = {100.0*(p_ok & anch).sum()/n_anch:.1f}%")
                delta = int((p_ok & anch).sum()) - int((b_ok & anch).sum())
                proj = 100.0 * (pred_ok.sum() + delta) / len(d)
                print(f"\n    => switching the anchor moves {delta:+d} problems")
                print(f"       shipped accuracy {100.0*pred_ok.mean():.2f}% "
                      f"-> projected {proj:.2f}%")
                print("       (exact for these rows: every other triage path is untouched)")

        print("\n  ORACLE CEILINGS")
        allc = [d[c].map(to_num).values for c in cand_cols]
        any_ok = np.zeros(len(d), dtype=bool)
        for v in allc:
            any_ok |= eq(v, gold)
        print(f"    shipped answer          : {100.0*pred_ok.mean():.2f}%")
        print(f"    oracle over candidates  : {100.0*any_ok.mean():.2f}%")
        print(f"    headroom left in select : {100.0*(any_ok.mean()-pred_ok.mean()):.2f}pp")

    # ---- grader sensitivity ------------------------------------------------
    if "correct" in d.columns:
        pred = d.predicted.map(to_num).values
        print("\n  GRADER SENSITIVITY (shipped answer)")
        print(f"    as graded in the run (abs 1e-3): {100.0*d.correct.mean():.2f}%")
        for rel in (1e-9, 1e-6, 1e-4):
            print(f"    relative {rel:<8g}             : {100.0*eq(pred, gold, rel).mean():.2f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for p in sys.argv[1:]:
        replay(p)
        print()
