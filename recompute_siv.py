"""
[v16.6] Step 1 — regenerate SIV's telemetry with an independent reference.

Every selector experiment in this project was fitted to telemetry produced by
`SIV.verify(blueprint, programmer_answer)`, where the Programmer wrote its code
FROM that blueprint. Layer 2 inverts the equation chain from the supplied
answer back to the givens, so feeding it an answer the chain itself produced
makes the inversion succeed by construction: `execution_rel_error` is exactly
zero on 81% of rows and `verified` is True on 20 of the 27 errors. The 2880
rules and two classifiers that all came out negative out-of-sample were fitted
to that.

The stored runs carry `blueprint_givens` and `blueprint_equations`, so the
audit can simply be re-run against a reference the blueprint did not produce.
This asks the question that was never actually asked: with telemetry that is
not vacuous, does a selector generalize?

Scope. The independent reference used here is the zero-shot baseline, because
it is the only non-coupled answer the finished runs stored. That makes this a
proof of principle, not a deployable rule: the production mechanism uses the
Architect's own expected_answer instead, so the MAS depends on nothing outside
itself, and expected_answer was never persisted -- which is exactly why this
has to be settled on a run rather than here.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

RUNS = {
    'v14.8': 'results_August/full_mas_20260819.csv',
    'v15.2': 'results_August/mas_full_20260824.csv',
    'v15.5': 'results_August/fullmas_20260830.csv',
    'v15.6': 'results_September/results/MAS_SHT/results/mas_sht_math7b_20260912_094311.csv',
}
OUT = 'recomputed_siv.csv'


def num(x) -> Optional[float]:
    try:
        v = float(x)
        return None if np.isnan(v) else v
    except (TypeError, ValueError):
        return None


def main() -> int:
    from siv_module import SymbolicInverseVerifier as SIV

    rows = []
    for run, path in RUNS.items():
        d = pd.read_csv(path)
        for i, r in d.iterrows():
            try:
                givens = json.loads(r.blueprint_givens)
                eqs = json.loads(r.blueprint_equations)
            except (json.JSONDecodeError, TypeError):
                givens, eqs = {}, []
            rec = dict(run=run, problem_id=r.problem_id, gold=num(r.gold),
                       base_ok=bool(r.baseline_correct), mas_ok=bool(r.correct),
                       # what the pipeline recorded (coupled reference)
                       old_verified=(str(r.siv_verified) == 'True'),
                       old_audit=(str(r.siv_execution_audit_passed) == 'True'),
                       old_conf=num(r.siv_confidence),
                       old_rel=num(r.siv_execution_rel_error),
                       old_matched=num(r.siv_givens_matched),
                       old_total=num(r.siv_givens_total))
            ref = num(r.baseline_ans)
            if givens and eqs and ref is not None:
                try:
                    res = SIV.verify({'givens': givens, 'equations': eqs}, ref)
                    rec.update(new_verified=bool(res.verified),
                               new_audit=bool(res.execution_audit_passed),
                               new_conf=res.confidence,
                               new_rel=res.execution_rel_error,
                               new_matched=res.givens_matched,
                               new_total=res.givens_total,
                               new_invertible=bool(res.invertible),
                               bp_answer=res.blueprint_answer)
                except Exception as exc:
                    rec['error'] = f"{type(exc).__name__}: {exc}"
            rows.append(rec)
            if (len(rows)) % 100 == 0:
                print(f"  ...{len(rows)}/600", flush=True)

    R = pd.DataFrame(rows)
    R.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}  ({len(R)} rows, "
          f"{int(R.get('new_verified', pd.Series(dtype=bool)).notna().sum())} recomputed)\n")

    ok = R[R.new_verified.notna()] if 'new_verified' in R else R.iloc[0:0]
    if not len(ok):
        print("nothing recomputed"); return 1

    print("=== IS THE TELEMETRY STILL VACUOUS? ===")
    print(f"  rows recomputed                     : {len(ok)}")
    print(f"  OLD rel_error exactly 0             : {(ok.old_rel <= 1e-12).mean():.1%}")
    print(f"  NEW rel_error exactly 0             : {(ok.new_rel <= 1e-12).mean():.1%}")
    print(f"  OLD audit passed                    : {ok.old_audit.mean():.1%}")
    print(f"  NEW audit passed                    : {ok.new_audit.mean():.1%}")
    print(f"  OLD verified                        : {ok.old_verified.mean():.1%}")
    print(f"  NEW verified                        : {ok.new_verified.mean():.1%}")

    print("\n=== DOES THE VERDICT SEPARATE RIGHT FROM WRONG? ===")
    print("  (a verdict that carries information should differ between the two)")
    for tag, col in [('OLD (coupled)', 'old_verified'), ('NEW (independent)', 'new_verified')]:
        t = ok.groupby(col).agg(n=('mas_ok', 'size'), mas_correct=('mas_ok', 'mean'))
        base = ok.mas_ok.mean()
        if True in t.index and False in t.index:
            sep = t.loc[True, 'mas_correct'] - t.loc[False, 'mas_correct']
        else:
            sep = float('nan')
        print(f"  {tag:<20} P(correct|verified)={t.loc[True,'mas_correct'] if True in t.index else float('nan'):.3f}  "
              f"P(correct|not)={t.loc[False,'mas_correct'] if False in t.index else float('nan'):.3f}  "
              f"separation={sep:+.3f}   (base rate {base:.3f})")
    return 0


if __name__ == '__main__':
    sys.exit(main())
