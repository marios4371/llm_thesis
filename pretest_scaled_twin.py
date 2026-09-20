"""
[v18.0] Pre-test: does a scaled twin let CoT read the structure it misreads
at 7 digits?

Pre-registered, written before the run:

    rows      the 40 gold-consistent seed-44 gsm-hard problems with a number
              >= 100k (pretest_data/clean_large_number_rows.json, self-
              contained: text, original text, gold, gold program)
    control   CoT alone on those rows, already measured: 32/40 = 80.0%
    arm S     twin = uniformly scaled copy (deployable)
    arm O     twin = the original GSM8K problem (oracle ceiling)

    O <= 33/40   magnitude is not what breaks the structure; close this.
    S >= 36/40   the mechanism works and is deployable; go to a full run.
    O >= 36 but S < 34
                 the mechanism works but the twin construction does not;
                 the remaining problem is building a coherent twin.

Cost: 80 CoT calls, one model, ~1 GPU hour. No baseline is re-run: the
control is the recorded result of the same model under greedy decoding on
the same text, which is deterministic.

    python pretest_scaled_twin.py                 # on Kaggle
    python pretest_scaled_twin.py --stub          # offline, fake model
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import scaled_twin as ST

MANIFEST = 'pretest_data/clean_large_number_rows.json'


def correct(pred: Optional[float], gold: Optional[float]) -> bool:
    if pred is None or gold is None:
        return False
    return abs(pred - gold) <= max(1e-3, 1e-4 * abs(gold))


class _StubClient:
    """Answers Problem A with the gold of a scaled twin (nonsense) and
    Problem B with the gold, so the whole script runs offline."""
    provider = "stub"
    model_name = "stub"

    def __init__(self, golds: Dict[str, float]):
        self.golds = golds
        self.calls = 0

    def call_model(self, msgs, **kw):
        self.calls += 1
        text = msgs[-1]["content"]
        gold = next((g for t, g in self.golds.items() if t.strip() in text), 0.0)
        return f"Problem A: ... Answer A: 7\nProblem B: ... Answer B: {gold}"


def build_client(preset: str, stub_golds: Optional[Dict[str, float]]):
    if stub_golds is not None:
        return _StubClient(stub_golds)
    from Mas_solver import AgentRole, UnifiedLLMClient, HETEROGENEOUS_PRESETS, SOLVER_VERSION
    mc = HETEROGENEOUS_PRESETS[preset][AgentRole.BASELINE]   # the CoT reader
    print(f"solver_version={SOLVER_VERSION} preset={preset}")
    print(f"CoT model: {mc.provider}/{mc.model_name} (4bit={getattr(mc, 'load_4bit', False)})")
    return UnifiedLLMClient(provider=mc.provider, use_cache=False,
                            model_override=mc.model_name,
                            load_4bit=getattr(mc, 'load_4bit', False))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default=MANIFEST)
    ap.add_argument('--preset', default='qwen_math7b_mixed')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default='pretest_scaled_twin.json')
    ap.add_argument('--max-tokens', type=int, default=1400)
    ap.add_argument('--stub', action='store_true', help='offline dry run')
    ap.add_argument('--skip-oracle', action='store_true')
    args = ap.parse_args()

    with open(args.manifest, encoding='utf-8') as fh:
        man = json.load(fh)
    rows: List[Dict] = man['rows'][:args.limit] if args.limit else man['rows']
    ctrl = sum(1 for r in rows if r['cot_correct'])
    print(f"rows: {len(rows)}  ({man.get('filter')})")
    print(f"control, CoT alone (recorded): {ctrl}/{len(rows)} = {100*ctrl/len(rows):.1f}%\n")

    client = build_client(args.preset,
                          {r['text']: r['gold'] for r in rows} if args.stub else None)

    out, t0 = [], time.time()
    for i, r in enumerate(rows, 1):
        rec = dict(pid=r['problem_id'], gold=r['gold'], max_number=r['max_number'],
                   cot_correct=bool(r['cot_correct']))
        # ---- arm S: uniformly scaled twin ---------------------------------
        sc = ST.scale_text(r['text'])
        rec['scaled_ok'] = sc.ok
        rec['divisor'] = sc.divisor
        if sc.ok:
            raw = client.call_model(
                [{"role": "user", "content": ST.twin_prompt(sc.twin_text, r['text'])}],
                temperature=0.0, max_tokens=args.max_tokens)
            a, b = ST.extract_answers(str(raw))
            rec.update(S_twin=sc.twin_text, S_answer_A=a, S_answer_B=b,
                       S_correct=correct(b, r['gold']), S_raw=str(raw)[-600:])
        else:
            rec.update(S_answer_B=None, S_correct=False, S_refused=sc.refused)
        # ---- arm O: the original problem as the twin ----------------------
        if not args.skip_oracle and r.get('original_text'):
            raw = client.call_model(
                [{"role": "user", "content": ST.twin_prompt(r['original_text'], r['text'])}],
                temperature=0.0, max_tokens=args.max_tokens)
            a, b = ST.extract_answers(str(raw))
            rec.update(O_answer_A=a, O_answer_B=b,
                       O_correct=correct(b, r['gold']), O_raw=str(raw)[-600:])
        out.append(rec)
        el = time.time() - t0
        print(f"[{i:2d}/{len(rows)}] {r['problem_id']:15s} "
              f"cot={'OK ' if rec['cot_correct'] else 'BAD'} "
              f"S={'OK ' if rec.get('S_correct') else 'BAD'} "
              f"O={'OK ' if rec.get('O_correct') else ('---' if args.skip_oracle else 'BAD')} "
              f"  {el/i:5.0f}s/row  eta {el/i*(len(rows)-i)/60:4.0f} min", flush=True)
        with open(args.out, 'w', encoding='utf-8') as fh:   # survive interruption
            json.dump(dict(manifest=args.manifest, preset=args.preset,
                           n=len(rows), rows=out), fh, ensure_ascii=False, indent=1)

    n = len(out)
    s = sum(1 for r in out if r.get('S_correct'))
    o = sum(1 for r in out if r.get('O_correct'))
    cov = sum(1 for r in out if r.get('scaled_ok'))
    print("\n" + "=" * 66)
    print(f"  control  CoT alone            {ctrl:2d}/{n} = {100*ctrl/n:5.1f}%")
    print(f"  arm S    scaled twin          {s:2d}/{n} = {100*s/n:5.1f}%   (twin built on {cov}/{n})")
    if not args.skip_oracle:
        print(f"  arm O    original as twin     {o:2d}/{n} = {100*o/n:5.1f}%   (oracle ceiling)")
    for arm in (['S'] + ([] if args.skip_oracle else ['O'])):
        w = sum(1 for r in out if r.get(f'{arm}_correct') and not r['cot_correct'])
        l = sum(1 for r in out if r['cot_correct'] and not r.get(f'{arm}_correct'))
        print(f"  arm {arm} vs CoT: W={w} L={l}")
    # the diagnostic: did the model even get the small twin right?
    sa = sum(1 for r in out if r.get('S_answer_A') is not None)
    print(f"\n  Problem A answered (scaled twin): {sa}/{n}; Problem B extracted: "
          f"{sum(1 for r in out if r.get('S_answer_B') is not None)}/{n}")
    print("\n  pre-registered reading:")
    if not args.skip_oracle and o <= 33:
        print("    O <= 33: magnitude is not what breaks the structure -> CLOSE")
    elif s >= 36:
        print("    S >= 36: mechanism works and is deployable -> FULL RUN")
    elif (not args.skip_oracle) and o >= 36 and s < 34:
        print("    O works, S does not: the twin construction is the problem")
    else:
        print("    inconclusive at n=40; report as measured")
    print(f"\n  saved: {args.out}   (download it — an interactive session loses it)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
