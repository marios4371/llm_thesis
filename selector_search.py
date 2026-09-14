"""
[v15.8] Run 0 -- is there a selector rule that generalizes?

The candidates the pipeline already generates contain the right answer on 56
of the 600 pooled rows where the baseline is wrong: a perfect selector would
score 85.50% against the baseline's 76.00%. The measured system captures
+0.17pp of that. So the question is not whether headroom exists, it is
whether any RULE can find it without also firing on the rows the baseline
already gets right -- the naive ">=2 candidates agree" rule scores 63.67%,
losing 114 rows to win 40, because candidates drawn from one model share
their errors.

Method. Every rule is scored leave-one-run-out: pick the best rule on three
runs, report what it does on the fourth, which that selection never saw. This
is the discipline v15.3 skipped -- its rule was derived on the same 150 rows
that measured it, read 76.00% in-sample, and did not replicate. The search is
deliberately over simple, interpretable conjunctions; with ~450 training rows
and a few thousand candidate rules, in-sample deltas are guaranteed to look
good, and only the held-out column means anything.
"""
from __future__ import annotations

import itertools
from collections import Counter
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binomtest

RUNS = {
    'v14.8': 'results_August/full_mas_20260819.csv',
    'v15.2': 'results_August/mas_full_20260824.csv',
    'v15.5': 'results_August/fullmas_20260830.csv',
    'v15.6': 'results_September/results/MAS_SHT/results/mas_sht_math7b_20260912_094311.csv',
}
CANDS = ['cand_primary', 'cand_blueprint_eval', 'cand_alt_1', 'cand_alt_2']


def num(x) -> Optional[float]:
    try:
        v = float(x)
        return None if np.isnan(v) else round(v, 6)
    except (TypeError, ValueError):
        return None


def eq(a, b, tol=1e-3) -> bool:
    return a is not None and b is not None and abs(a - b) < tol


def build() -> pd.DataFrame:
    """One row per problem, with the challenger each strategy would propose."""
    out = []
    for run, path in RUNS.items():
        d = pd.read_csv(path)
        for _, r in d.iterrows():
            gold = num(r.gold)
            base = num(r.baseline_ans)
            if gold is None:
                continue
            vals = [num(r[c]) for c in CANDS]
            present = [v for v in vals if v is not None]
            cnt = Counter(present)

            maj = (max(cnt.items(), key=lambda kv: (kv[1], -kv[0]))[0]
                   if cnt else None)
            ch = {'majority': maj, 'primary': vals[0], 'blueprint_eval': vals[1]}
            na = {k: (cnt.get(v, 0) if v is not None else 0) for k, v in ch.items()}

            gm, gt = num(r.siv_givens_matched), num(r.siv_givens_total)
            out.append(dict(
                run=run, problem_id=r.problem_id, gold=gold, base=base,
                base_ok=bool(r.baseline_correct), mas_ok=bool(r.correct),
                ch_majority=ch['majority'], ch_primary=ch['primary'],
                ch_blueprint_eval=ch['blueprint_eval'],
                na_majority=na['majority'], na_primary=na['primary'],
                na_blueprint_eval=na['blueprint_eval'],
                n_distinct=len(cnt), n_present=len(present),
                siv_verified=(str(r.siv_verified) == 'True'),
                siv_invertible=(str(r.siv_invertible) == 'True'),
                siv_taut=(str(r.siv_tautological) == 'True'),
                siv_audit=(str(r.siv_execution_audit_passed) == 'True'),
                siv_conf=num(r.siv_confidence) or 0.0,
                givens_ratio=(gm / gt if gm is not None and gt else 0.0),
                givens_total=gt or 0.0,
                prov=str(r.blueprint_provenance),
                repaired=(str(r.blueprint_repaired) == 'True'),
                ver_passed=(str(r.verification_passed) == 'True'),
                ver_conf=num(r.verification_confidence) or 0.0,
                triggered=(str(r.sht_triggered) == 'True'),
            ))
    return pd.DataFrame(out)


# Kept small and readable on purpose: every extra predicate multiplies the
# search space and buys more in-sample luck, which the held-out column then
# has to strip back out.
PREDICATES: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    'siv_verified':       lambda f: f.siv_verified,
    'not_siv_verified':   lambda f: ~f.siv_verified,
    'siv_invertible':     lambda f: f.siv_invertible,
    'not_tautological':   lambda f: ~f.siv_taut,
    'siv_audit_passed':   lambda f: f.siv_audit,
    'all_givens_matched': lambda f: f.givens_ratio >= 0.999,
    'givens_ge_2':        lambda f: f.givens_total >= 2,
    'givens_ge_3':        lambda f: f.givens_total >= 3,
    'prov_primary_json':  lambda f: f.prov == 'primary_json',
    'not_repaired':       lambda f: ~f.repaired,
    'verifier_passed':    lambda f: f.ver_passed,
    'ver_conf_ge_08':     lambda f: f.ver_conf >= 0.8,
    'siv_conf_ge_08':     lambda f: f.siv_conf >= 0.8,
    'sht_triggered':      lambda f: f.triggered,
    'unanimous_cands':    lambda f: f.n_distinct <= 1,
}


def apply_rule(f: pd.DataFrame, strat: str, k: int,
               preds: Tuple[str, ...]) -> pd.Series:
    """Boolean per row: does this rule override the baseline?"""
    ch = f['ch_' + strat]
    fires = (f['na_' + strat] >= k) & ch.notna()
    # an "override" that reproduces the baseline is not an override
    same = pd.Series([eq(c, b) for c, b in zip(ch, f.base)], index=f.index)
    fires &= ~same
    for p in preds:
        fires &= PREDICATES[p](f)
    return fires


def score(f: pd.DataFrame, fires: pd.Series, strat: str) -> Dict[str, float]:
    ch = f['ch_' + strat]
    ch_ok = np.array([eq(c, g) for c, g in zip(ch, f.gold)])
    picked_ok = np.where(fires.values, ch_ok, f.base_ok.values)
    w = int((picked_ok & ~f.base_ok.values).sum())
    l = int((~picked_ok & f.base_ok.values).sum())
    return dict(acc=float(picked_ok.mean()),
                delta=float(picked_ok.mean() - f.base_ok.mean()),
                W=w, L=l, fired=int(fires.sum()))


def rules():
    names = list(PREDICATES)
    for strat in ('majority', 'primary', 'blueprint_eval'):
        for k in (1, 2, 3):
            if strat != 'majority' and k > 1:
                continue   # n_agree on a fixed candidate is just its own count
            for size in (0, 1, 2, 3):
                for combo in itertools.combinations(names, size):
                    yield strat, k, combo


def main() -> None:
    f = build()
    print(f"pooled rows: {len(f)}   baseline {f.base_ok.mean():.4f}   "
          f"MAS {f.mas_ok.mean():.4f}")
    all_rules = list(rules())
    print(f"rule space: {len(all_rules)}\n")

    fold_rows = []
    for held in RUNS:
        tr, te = f[f.run != held], f[f.run == held]
        best, best_d = None, -9.0
        for strat, k, preds in all_rules:
            s = score(tr, apply_rule(tr, strat, k, preds), strat)
            if s['delta'] > best_d:
                best, best_d = (strat, k, preds), s['delta']
        strat, k, preds = best
        s_tr = score(tr, apply_rule(tr, strat, k, preds), strat)
        s_te = score(te, apply_rule(te, strat, k, preds), strat)
        fold_rows.append(dict(
            held_out=held,
            rule=f"{strat}>={k} & " + (" & ".join(preds) or "-"),
            train_delta=s_tr['delta'], test_delta=s_te['delta'],
            test_W=s_te['W'], test_L=s_te['L'], test_fired=s_te['fired']))

    R = pd.DataFrame(fold_rows)
    pd.set_option('display.width', 220)
    pd.set_option('display.max_colwidth', 68)
    print("=== LEAVE-ONE-RUN-OUT: best rule picked on 3 runs, scored on the 4th ===")
    print(R.to_string(index=False))
    print(f"\nmean held-out delta vs baseline: {R.test_delta.mean()*100:+.2f}pp")
    W, L = int(R.test_W.sum()), int(R.test_L.sum())
    p = binomtest(W, W + L, 0.5).pvalue if W + L else float('nan')
    print(f"pooled held-out: W={W} L={L}  McNemar p={p:.4f}")
    print(f"reference: the measured MAS is "
          f"{(f.mas_ok.mean()-f.base_ok.mean())*100:+.2f}pp over the same 600 rows")


if __name__ == '__main__':
    main()
