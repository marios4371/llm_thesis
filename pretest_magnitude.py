"""
[v16.2] Targeted pre-test: does solving a SHRUNK copy beat solving the original?

Pre-registered criterion, written before the run:
    shrink route > 75% on the large-number rows  -> the mechanism works, go to
                                                    a full run
    shrink route <= 65% (i.e. no better than PAL) -> shrinking adds nothing,
                                                    close it

Why these numbers. On the 54 seed-44 rows whose largest number is at least
100k, the measured scores are baseline 51.9%, MAS 55.6%, PAL 64.8%; the same
model reaches 98.4% on gsm-hard rows whose numbers are under 100. PAL's +13pp
over the baseline already confirms that delegating arithmetic recovers part of
the collapse, and the remaining ~33pp is attributed to transcribing 8-digit
operands into the program -- which is what shrinking removes.

The design keeps ONE variable. Both arms call the SAME Architect through the
SAME entry point and evaluate through the SAME CAS; the only difference is
whether the problem text it reads carries 3473626 or 7. So a gap between the
arms cannot be a prompt effect, a parser effect or an arithmetic-backend
effect -- it is magnitude.

Cost: only the large-number rows, one Architect call per arm, one model
loaded. ~1 GPU hour.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import magnitude_invariance as M

# The pre-test reads a committed manifest, not the run CSV and not HuggingFace.
# On the runner neither exists: .gitignore excludes results/ so the CSV never
# reaches a fresh clone, and a Kaggle session has no dataset cache, so forcing
# HF_DATASETS_OFFLINE there fails every load. The manifest carries the problem
# TEXT inline (54 rows, 22 KB), which makes the run self-contained -- no
# download, no cache, no network -- and pins it to exactly the rows the
# analysis was done on rather than re-deriving them from a seed.
MANIFEST = 'pretest_data/large_number_rows.json'
DEFAULT_RUN = ('results_September/results/MAS_SHT/results/'
               'mas_sht_math7b_20260912_094311.csv')


def num(x) -> Optional[float]:
    try:
        v = float(x)
        return None if v != v else v
    except (TypeError, ValueError):
        return None


def correct(pred: Optional[float], gold: Optional[float]) -> bool:
    if pred is None or gold is None:
        return False
    return abs(pred - gold) <= 1e-6 * max(abs(gold), 1.0) or abs(pred - gold) < 1e-3


def load_rows(manifest: str, run_csv: str,
              min_magnitude: float) -> List[Dict]:
    """Prefer the committed manifest; fall back to the CSV + HF cache locally."""
    if os.path.isfile(manifest):
        with open(manifest, encoding='utf-8') as fh:
            man = json.load(fh)
        rows = [r for r in man['rows'] if r['mx'] >= min_magnitude]
        print(f"manifest: {manifest}  (source run {man.get('source_run')}, "
              f"seed {man.get('seed')})")
        return [dict(pid=r['problem_id'], text=r['text'], gold=r['gold'],
                     mx=r['mx'], mas_correct=r['mas_correct'],
                     baseline_correct=r['baseline_correct']) for r in rows]

    print(f"manifest not found at {manifest} -- falling back to {run_csv} + HF cache")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    import pandas as pd
    import grounding_probe as G
    texts = G.load_texts()
    d = pd.read_csv(run_csv)
    out = []
    for _, r in d.iterrows():
        t = texts.get(r.problem_id)
        if not t:
            continue
        nums = [abs(x) for x in M.text_number_set(t)]
        mx = max(nums) if nums else 0.0
        if mx < min_magnitude:
            continue
        out.append(dict(pid=r.problem_id, text=t, gold=num(r.gold), mx=mx,
                        mas_correct=bool(r.correct),
                        baseline_correct=bool(r.baseline_correct)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default=MANIFEST)
    ap.add_argument('--run-csv', default=DEFAULT_RUN)
    ap.add_argument('--min-magnitude', type=float, default=1e5)
    ap.add_argument('--preset', default='qwen_math7b_mixed')
    ap.add_argument('--limit', type=int, default=0, help='0 = all')
    ap.add_argument('--out', default='pretest_magnitude.json')
    args = ap.parse_args()

    rows = load_rows(args.manifest, args.run_csv, args.min_magnitude)
    if args.limit:
        rows = rows[:args.limit]
    print(f"large-number rows (max >= {args.min_magnitude:g}): {len(rows)}")

    from Mas_solver import (AgentRole, UnifiedLLMClient,
                            QualityEnhancedMultiAgentSolver,
                            HETEROGENEOUS_PRESETS, SOLVER_VERSION)
    mc = HETEROGENEOUS_PRESETS[args.preset][AgentRole.MATHEMATICIAN]
    print(f"solver_version={SOLVER_VERSION} preset={args.preset}")
    print(f"Architect: {mc.provider}/{mc.model_name} "
          f"(4bit={getattr(mc, 'load_4bit', False)})\n")
    client = UnifiedLLMClient(provider=mc.provider, use_cache=False,
                              model_override=mc.model_name,
                              load_4bit=getattr(mc, 'load_4bit', False))
    solver = QualityEnhancedMultiAgentSolver(
        clients={role: client for role in AgentRole})

    def architect(text: str) -> Dict:
        try:
            return solver.run_mathematician_analysis(text) or {}
        except Exception as exc:
            print(f"      architect raised {type(exc).__name__}: {exc}")
            return {}

    out, t0 = [], time.time()
    for i, r in enumerate(rows, 1):
        sh = M.shrink_text(r['text'])
        rec = dict(pid=r['pid'], gold=r['gold'], mx=r['mx'],
                   mas_correct=r['mas_correct'],
                   baseline_correct=r['baseline_correct'],
                   n_shrunk=sh.n_shrunk)

        # ARM A -- the original problem, large numbers intact
        bp_big = architect(r['text'])
        a = M.evaluate(bp_big)
        rec['direct_answer'] = a
        rec['direct_correct'] = correct(a, r['gold'])

        # ARM B -- the same Architect on the shrunk copy, then rebind
        bp_small = architect(sh.shrunk_text)
        rec['shrunk_answer_at_probes'] = M.evaluate(bp_small)
        reb, ok, why = M.rebind_guarded(bp_small.get('givens') or {}, sh)
        rec['rebind_ok'], rec['rebind_reason'] = ok, why
        b = M.evaluate({**bp_small, 'givens': reb}) if ok else None
        rec['shrink_answer'] = b
        rec['shrink_correct'] = correct(b, r['gold'])

        # metamorphic agreement between the two structures, at the probes
        mm = M.metamorphic_check(bp_big, bp_small, sh)
        rec['metamorphic_agree'] = mm.agree
        rec['metamorphic_rel_error'] = mm.rel_error

        out.append(rec)
        print(f"  [{i}/{len(rows)}] {r['pid']:<18} gold={r['gold']!s:<16} "
              f"direct={'OK ' if rec['direct_correct'] else 'BAD'} "
              f"shrink={'OK ' if rec['shrink_correct'] else 'BAD'} "
              f"agree={rec['metamorphic_agree']}"
              + ('' if ok else f"  [abstained: {why[:60]}]"))

    n = max(1, len(out))
    direct = sum(r['direct_correct'] for r in out)
    shrink = sum(r['shrink_correct'] for r in out)
    mas = sum(r['mas_correct'] for r in out)
    base = sum(r['baseline_correct'] for r in out)
    covered = [r for r in out if r['rebind_ok']]

    print("\n" + "=" * 72)
    print(f"LARGE-NUMBER ROWS  n={n}   ({time.time()-t0:.0f}s)")
    print("=" * 72)
    print(f"  baseline (from the run)      {base}/{n} = {100*base/n:5.1f}%")
    print(f"  MAS      (from the run)      {mas}/{n} = {100*mas/n:5.1f}%")
    print(f"  ARM A blueprint, as-is       {direct}/{n} = {100*direct/n:5.1f}%")
    print(f"  ARM B blueprint, SHRUNK      {shrink}/{n} = {100*shrink/n:5.1f}%"
          f"   <-- the pre-registered number")
    print(f"\n  coverage (rebind allowed)    {len(covered)}/{n} = "
          f"{100*len(covered)/n:5.1f}%")
    if covered:
        cc = sum(r['shrink_correct'] for r in covered)
        print(f"  ARM B where it did answer    {cc}/{len(covered)} = "
              f"{100*cc/len(covered):5.1f}%")
    w = sum(1 for r in out if r['shrink_correct'] and not r['direct_correct'])
    l = sum(1 for r in out if r['direct_correct'] and not r['shrink_correct'])
    print(f"\n  shrink vs direct: W={w} L={l}")
    print(f"  PRE-REGISTERED: >75% proceed to a full run; <=65% close it.")

    with open(args.out, 'w', encoding='utf-8') as fh:
        json.dump(dict(solver_version=str(SOLVER_VERSION), n=n,
                       min_magnitude=args.min_magnitude, rows=out), fh, indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
