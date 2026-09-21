"""
[v19.0] Pre-test: does a WELL-POSED small twin let the model reason where it
can read, when the construction no longer breaks the problem?

WHY THIS IS NOT "another shrink variant"
----------------------------------------
v16.2 tested exactly this mechanism and was rejected by its pre-registered
criterion (shrunk blueprint 57.4% against a 65% floor; shrink vs direct
W=5 L=9), and the note in memory says: do not spend more GPU on shrink
variants. That instruction was right for the thing it was written about. The
same log, however, names the two causes:

    "Order inversion is NOT the cause [...] It is granularity and feasibility."
    root beer 25956168 -> 11 with 5956168 -> 13 gives "drink 13 from 11"
    5611617 riders -> 7 with 25% upright gives 1.75 people

Both are properties of the CONSTRUCTION, not of the mechanism, and both have
now been measured over the 754 gold-consistent gsm-hard rows with a number
>= 1e5 and closed in `scaled_twin.scale_text`:

    defect                                        v16.2-style    v19
    a quantity in the twin goes through zero         5.4%        0.0%
    a whole quantity becomes fractional              9.9%        2.3%
    twin values' median distance from exact           --         2.0%

(The v16.2 probes were independent primes, so its own rates are worse than the
5.4%/9.9% measured here for a uniform rescaling; those are the rates for the
gentler construction this replaces.)

Everything downstream is v16.2's, unchanged and reused through
`scaled_twin.as_shrink_result`: the guarded rebind, whose guard covered 85.2%
of rows and "worked as designed", and CAS evaluation through the shipped SIV.
One thing changed: the twin.

THE ARMS
--------
Control   plain CoT on the real problem. Recorded, greedy, 32/40 on this set.

  W  warm-up.    One call. The twin and the original are shown together and the
                 model solves the twin first, then the original "with exactly
                 the same steps". The model still reads the large numbers, but
                 with the structure already fixed in context.

  T  transplant. One call. The model is shown ONLY the twin and returns a
                 blueprint. The real values are rebound deterministically and
                 the answer is computed by the CAS. The model never sees a
                 seven-digit number, so it cannot misread one. Where the guard
                 refuses -- a scaled input never reached the givens, so the
                 real value cannot be restored -- the row falls back to plain
                 CoT, which is what a deployed system would do.

  O  oracle.     One call, W's prompt with the ORIGINAL GSM8K problem as the
                 twin (gsm-hard keeps it in the gold program's docstring). Not
                 deployable; it is the ceiling, and it is what tells a failure
                 of the mechanism apart from a failure of the construction.

PRE-REGISTERED, written before the run
--------------------------------------
    T >= 36/40   the transplant works and is deployable -> full run
    W >= 36/40   the warm-up works and is deployable    -> full run
    O <= 33/40   even a perfect twin does not help; magnitude is not what
                 breaks the structure here, and this whole direction closes
    O >= 36 but both W and T <= 33
                 the mechanism is real and the construction is still what
                 stands in the way

Anything else is reported as measured, with no verdict. T's accuracy is
reported twice: over all 40 rows (with the CoT fallback, which is the
deployable number) and over the rows the guard covered (which is the
mechanism's own number). Cost: <=120 calls, ~1.5 GPU hours.

One note on grading, in T's disfavour. This repo scores an answer correct
within `max(1e-3, 1e-4 * |gold|)`, a RELATIVE tolerance, so on a gold of
3,473,609 anything within +-347 passes. Arm T computes with a CAS and cannot
make an arithmetic slip; arms W, O and the control let the model do the
arithmetic and are forgiven their near misses. The comparison is therefore
conservative for T, and a transplant that wins here wins with the scoring
tilted against it.

    python pretest_v19.py                    # on Kaggle
    python pretest_v19.py --stub             # offline, no model
    python pretest_v19.py --arms WT          # skip the oracle
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Dict, List, Optional

import scaled_twin as ST

MANIFEST = 'pretest_data/clean_large_number_rows.json'


def correct(pred: Optional[float], gold: Optional[float]) -> bool:
    if pred is None or gold is None:
        return False
    return abs(pred - gold) <= max(1e-3, 1e-4 * abs(gold))


# ---------------------------------------------------------------------------
# offline stubs -- they exercise the real rebind and the real CAS
# ---------------------------------------------------------------------------

class _StubClient:
    provider = model_name = "stub"

    def __init__(self, golds: Dict[str, float]):
        self.golds = golds
        self.calls = 0

    def call_model(self, msgs, **kw):
        self.calls += 1
        text = msgs[-1]["content"]
        gold = next((g for t, g in self.golds.items() if t.strip() in text), 0.0)
        return f"Problem A: ...\nAnswer A: 7\nProblem B: ...\nAnswer B: {gold}"


class _StubSolver:
    """Returns the twin's largest number as a single given, so the rebind and
    the CAS run for real: a correct transplant must hand back the ORIGINAL
    large number, which is what test_v19 asserts."""

    def run_mathematician_analysis(self, text, **kw):
        vals = []
        for tok in ST.NUMBER_RE.findall(text):
            try:
                vals.append(float(tok.replace(",", "")))
            except ValueError:
                pass
        return {"givens": {"v0": max(vals) if vals else 0.0},
                "equations": ["result = givens['v0']"]}


def build(preset: str, stub_golds, arms: List[str]):
    """(client for W/O, solver for T). Roles keep their production models.

    Only what the requested arms need is loaded. W and O read with the BASELINE
    model, T writes blueprints with the MATHEMATICIAN model, and in the shipped
    preset those are two different 7B checkpoints -- so `--arms WO` and
    `--arms T` each hold one model, which is the way out if both together do
    not fit in VRAM.
    """
    if stub_golds is not None:
        return _StubClient(stub_golds), _StubSolver()
    from Mas_solver import (AgentRole, UnifiedLLMClient, HETEROGENEOUS_PRESETS,
                            QualityEnhancedMultiAgentSolver, SOLVER_VERSION)
    cfg = HETEROGENEOUS_PRESETS[preset]
    bc, mc = cfg[AgentRole.BASELINE], cfg[AgentRole.MATHEMATICIAN]
    need_baseline = any(a in arms for a in ('W', 'O'))
    need_math = 'T' in arms
    print(f"solver_version={SOLVER_VERSION} preset={preset}")
    if need_baseline:
        print(f"  CoT reader   (arms W/O): {bc.provider}/{bc.model_name}")
    if need_math:
        print(f"  Architect    (arm  T  ): {mc.provider}/{mc.model_name}")

    def mk(m):
        return UnifiedLLMClient(provider=m.provider, use_cache=False,
                                model_override=m.model_name,
                                load_4bit=getattr(m, 'load_4bit', False))

    baseline_client = mk(bc) if need_baseline else None
    if not need_math:
        return baseline_client, None
    math_client = (baseline_client if (baseline_client is not None
                                       and mc.model_name == bc.model_name)
                   else mk(mc))
    fallback = baseline_client or math_client
    solver = QualityEnhancedMultiAgentSolver(
        clients={r: (math_client if r == AgentRole.MATHEMATICIAN else fallback)
                 for r in AgentRole})
    return baseline_client, solver


# ---------------------------------------------------------------------------
# arm T: solve the twin, rebind the real values, compute exactly
# ---------------------------------------------------------------------------

def transplant(solver, sc: ST.ScaleResult) -> Dict:
    """One call on the twin; everything after it is deterministic."""
    import magnitude_invariance as MI
    out: Dict[str, object] = {}
    try:
        bp = solver.run_mathematician_analysis(sc.twin_text) or {}
    except Exception as exc:
        return {"answer": None, "abstain": f"architect raised {type(exc).__name__}: {exc}"}
    givens = bp.get("givens") or {}
    equations = bp.get("equations") or []
    out["givens_twin"] = givens
    out["equations"] = equations
    if not givens or not equations:
        out.update(answer=None, abstain="the twin produced no usable blueprint")
        return out
    rebound, ok, reason = MI.rebind_guarded(givens, ST.as_shrink_result(sc))
    out["rebind_reason"] = reason
    out["givens_rebound"] = rebound
    if not ok:
        out.update(answer=None, abstain=reason)
        return out
    value = MI.evaluate({**bp, "givens": rebound})
    out["answer"] = value
    if value is None:
        out["abstain"] = "the rebound blueprint did not evaluate"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default=MANIFEST)
    ap.add_argument('--preset', default='qwen_math7b_mixed')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default='pretest_v19.json')
    ap.add_argument('--max-tokens', type=int, default=1400)
    ap.add_argument('--arms', default='WTO', help='any of W, T, O')
    ap.add_argument('--stub', action='store_true')
    args = ap.parse_args()
    arms = [a for a in 'WTO' if a in args.arms.upper()]

    with open(args.manifest, encoding='utf-8') as fh:
        man = json.load(fh)
    rows: List[Dict] = man['rows'][:args.limit] if args.limit else man['rows']
    ctrl = sum(1 for r in rows if r['cot_correct'])
    print(f"rows: {len(rows)}  ({man.get('filter')})")
    print(f"arms: {' '.join(arms)}")
    print(f"control, CoT alone (recorded): {ctrl}/{len(rows)} = {100*ctrl/len(rows):.1f}%\n")

    client, solver = build(args.preset,
                           {r['text']: r['gold'] for r in rows} if args.stub else None,
                           arms)

    out, t0 = [], time.time()
    for i, r in enumerate(rows, 1):
        gold = r['gold']
        rec = dict(pid=r['problem_id'], gold=gold, max_number=r['max_number'],
                   cot_correct=bool(r['cot_correct']))
        sc = ST.scale_text(r['text'])
        rec.update(twin_ok=sc.ok, divisor=sc.divisor, smooth=sc.smooth,
                   twin=sc.twin_text if sc.ok else None)

        if 'W' in arms:
            if sc.ok:
                raw = client.call_model(
                    [{"role": "user", "content": ST.twin_prompt(sc.twin_text, r['text'])}],
                    temperature=0.0, max_tokens=args.max_tokens)
                a, b = ST.extract_answers(str(raw))
                rec.update(W_answer_A=a, W_answer=b, W_correct=correct(b, gold),
                           W_raw=str(raw)[-500:])
            else:
                rec.update(W_answer=None, W_correct=bool(r['cot_correct']),
                           W_fallback=True, W_abstain=sc.refused)

        if 'T' in arms:
            if sc.ok:
                t = transplant(solver, sc)
                rec.update(T_answer=t.get("answer"),
                           T_givens_twin=t.get("givens_twin"),
                           T_givens_rebound=t.get("givens_rebound"),
                           T_equations=t.get("equations"),
                           T_rebind=t.get("rebind_reason"),
                           T_abstain=t.get("abstain"))
                if t.get("answer") is None:
                    rec.update(T_covered=False, T_correct=bool(r['cot_correct']),
                               T_fallback=True)
                else:
                    rec.update(T_covered=True,
                               T_correct=correct(t["answer"], gold),
                               T_fallback=False)
            else:
                rec.update(T_answer=None, T_covered=False, T_fallback=True,
                           T_correct=bool(r['cot_correct']), T_abstain=sc.refused)

        if 'O' in arms and r.get('original_text'):
            raw = client.call_model(
                [{"role": "user", "content": ST.twin_prompt(r['original_text'], r['text'])}],
                temperature=0.0, max_tokens=args.max_tokens)
            a, b = ST.extract_answers(str(raw))
            rec.update(O_answer_A=a, O_answer=b, O_correct=correct(b, gold),
                       O_raw=str(raw)[-500:])

        out.append(rec)
        el = time.time() - t0
        flags = ' '.join(f"{k}={'OK ' if rec.get(f'{k}_correct') else 'BAD'}" for k in arms)
        print(f"[{i:2d}/{len(rows)}] {r['problem_id']:15s} "
              f"cot={'OK ' if rec['cot_correct'] else 'BAD'} {flags}"
              f"  {el/i:5.0f}s/row  eta {el/i*(len(rows)-i)/60:4.0f} min", flush=True)
        with open(args.out, 'w', encoding='utf-8') as fh:
            json.dump(dict(manifest=args.manifest, preset=args.preset, arms=arms,
                           n=len(rows), control=ctrl, rows=out), fh,
                      ensure_ascii=False, indent=1)

    # ---------------- summary -------------------------------------------
    n = len(out)
    score = {a: sum(1 for r in out if r.get(f'{a}_correct')) for a in arms}
    print("\n" + "=" * 70)
    print(f"  control  CoT alone              {ctrl:2d}/{n} = {100*ctrl/n:5.1f}%")
    names = {'W': 'warm-up twin', 'T': 'transplant  ', 'O': 'oracle twin '}
    for a in arms:
        print(f"  arm {a}    {names[a]}          {score[a]:2d}/{n} = {100*score[a]/n:5.1f}%")
    if 'T' in arms:
        cov = [r for r in out if r.get('T_covered')]
        hit = sum(1 for r in cov if r.get('T_correct'))
        print(f"           transplant on the rows it covered: {hit}/{len(cov)}"
              + (f" = {100*hit/len(cov):5.1f}%" if cov else "")
              + f"   (guard abstained on {n-len(cov)})")
    for a in arms:
        w = sum(1 for r in out if r.get(f'{a}_correct') and not r['cot_correct'])
        l = sum(1 for r in out if r['cot_correct'] and not r.get(f'{a}_correct'))
        print(f"  arm {a} vs CoT: W={w} L={l}")

    if n != len(man['rows']):
        print(f"\n  (n={n}, not the pre-registered {len(man['rows'])} -- no verdict)")
    else:
        print("\n  pre-registered reading:")
        O, W, T = score.get('O'), score.get('W'), score.get('T')
        if O is not None and O <= 33:
            print("    O <= 33: even a perfect twin does not help -> CLOSE this direction")
        elif T is not None and T >= 36:
            print("    T >= 36: the transplant works and is deployable -> FULL RUN")
        elif W is not None and W >= 36:
            print("    W >= 36: the warm-up works and is deployable -> FULL RUN")
        elif O is not None and O >= 36 and max(x for x in (W, T) if x is not None) <= 33:
            print("    O works, W/T do not: the construction is still the obstacle")
        else:
            print("    inconclusive at n=40; report as measured")
    print(f"\n  saved: {args.out}   (download it -- an interactive session loses it)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
