"""
[v20.0] Pre-test: does diversity in the INPUT space beat diversity in the
output space, at the same number of calls?

WHY THIS AND NOT ANOTHER SELECTION LAYER
----------------------------------------
Because the selection route is closed by measurement, and because the number
it was being measured against was mostly noise. Two findings, 2026-09-22:

  * On the 40 gold-consistent large-number rows, CoT 80.0%, PAL 80.0%,
    v17.1 82.5%, ORACLE(CoT, PAL) 85.0%. Both solvers are wrong on the same
    6 rows, and at least 4 of those 6 are still defective golds. True accuracy
    is ~90% with a true oracle near 92.5%: **there is ~2.5pp of selection
    headroom left on gsm-hard, so no aggregation over answers can show an
    effect there.** That is why v12 through v19 all measured null.
  * A mechanical audit of GSM-Hard's construction puts its defect rate at
    21.3% of the auditable rows, through three channels independent of the
    earlier 17.2% finding, one of them GSM8K-Platinum's human re-annotation.
    See gsmhard_audit.py.

So this pre-test does not run on gsm-hard. It runs where the model still has
somewhere to fall: gsm-symbolic-p2 or gsm-plus.

The mechanism is stated in renderings.py. In one line: a multi-agent system on
one local model with `do_sample=False` has no source of diversity at all, and
input-space re-rendering is a source that the robustness literature has
measured for years and never spent.

THE ARMS
--------
Control  C   plain greedy CoT on the problem as given. One call. Its answer is
             also the `identity` rendering's vote, so the ensemble never pays
             twice for it.

  F  form-diverse. Majority over renderings of the SAME problem, each solved
             once, greedily. Structural renderings are free; `paraphrase`
             costs one presenter call. A rendering that does not preserve the
             problem's numeric literals exactly is refused and falls back to
             the identity, so a rendering can never vote on a problem the
             model was not shown.

  S  iso-compute self-consistency. The same solver, same call budget as F's
             solver calls, on the ORIGINAL problem, sampled at temperature.
             **This arm decides the result.** Input-space diversity has to beat
             output-space diversity at equal cost, or it is just self-
             consistency with extra steps. It needs sampling, which this repo
             has had hardcoded off since v10.3; `Mas_solver.LOCAL_HF_SAMPLING`
             turns it on for this arm only and the run probes it before
             spending an hour on it.

PRE-REGISTERED, written before the run (n=100)
----------------------------------------------
Paired, against the control, on the same rows:

    F - C >= +6 rows and F's wins >= 2x its losses
                     input-space diversity pays -> full run
    F <= C           it does not pay -> this direction closes
    F >= S and F's wins over S >= its losses + 4
                     the claim of the thesis holds: the gain is from the
                     re-rendering and not from spending more calls
    mean disagreement(F renderings) <= mean disagreement(S samples)
                     KILL, whatever the accuracies say: re-rendering is not
                     producing more diversity than temperature, so there is no
                     mechanism to write up

n=100 is a pre-test, not a confirmation: with ~20 discordant pairs it detects
a large effect and nothing subtle. It is sized to decide whether to spend a
full run, and the summary reports McNemar's discordant counts rather than a
p-value so nobody reads it as one.

Even a null publishes. "Input-space diversity does not decorrelate more than
temperature sampling in multi-step reasoning" has never been measured; the two
closest papers (ParaMAWPS, PCS) each name multi-step reasoning as future work.

COST
----
7 calls/problem at 45.8 s/call measured -> ~8.9 h at n=100. Split it:

    python pretest_v20.py --build-manifest --dataset gsm-symbolic-p2 --n 100
    python pretest_v20.py --arms CF      # session 1, 4 calls/problem, ~5.1 h
    python pretest_v20.py --arms S       # session 2, 3 calls/problem, ~3.8 h
    python pretest_v20.py --stub         # offline, no model, no GPU

Sessions 1 and 2 read the same manifest and merge into the same output file,
so the arms stay paired across sessions. All three arms in one 9-hour session
(`--arms CFS`) is the simpler plan when the session length allows it, and it
removes the need to carry the JSON between sessions at all.

Re-running the same command RESUMES: a row already carrying every requested
arm is skipped. That matters because this run is long and this project has
lost long runs to a session dying partway. Download the JSON after every
session anyway -- an interactive Kaggle session loses /kaggle/working unless
the notebook is saved as a version.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Dict, List, Optional

import renderings as R

MANIFEST = 'pretest_data/v20_problems.json'
OUT = 'pretest_v20.json'


def correct(pred: Optional[float], gold: Optional[float]) -> bool:
    """The repo's grading tolerance, unchanged."""
    if pred is None or gold is None:
        return False
    return abs(pred - gold) <= max(1e-3, 1e-4 * abs(gold))


# ---------------------------------------------------------------------------
# manifest -- built once, offline, so both sessions score the same rows
# ---------------------------------------------------------------------------

def _gold(raw) -> Optional[float]:
    s = str(raw)
    if '####' in s:
        s = s.split('####')[-1]
    s = s.strip().replace(',', '').replace('$', '')
    try:
        return float(s.split()[0]) if s.split() else None
    except ValueError:
        return None


def build_manifest(dataset: str, n: int, seed: int, path: str) -> int:
    """Sample the rows once and write them down.

    Datasets are chosen for headroom, not for continuity with earlier runs:
    an arm cannot show an effect on a set the model already solves. gsm-hard
    is deliberately not an option here.
    """
    from Mas_solver import EnhancedProblemManager
    mgr = EnhancedProblemManager(random_seed=seed)
    pool = mgr.load_random_problems([dataset], n * 2)
    rows = []
    for it in pool:
        g = _gold(it.get('answer'))
        text = (it.get('puzzle') or '').strip()
        if g is None or len(text) < 20:
            continue
        rows.append({'problem_id': it.get('id'), 'text': text, 'gold': g,
                     'dataset': it.get('dataset')})
        if len(rows) >= n:
            break
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump({'dataset': dataset, 'seed': seed, 'n': len(rows),
                   'note': 'v20 pre-test rows. Built offline; both sessions '
                           'read this file so the arms stay paired.',
                   'rows': rows}, fh, ensure_ascii=False, indent=1)
    print(f"wrote {len(rows)} rows from {dataset} (seed {seed}) -> {path}")
    if len(rows) < n:
        print(f"  WARNING: asked for {n}, the loader yielded {len(rows)} usable")
    return 0 if rows else 1


# ---------------------------------------------------------------------------
# offline stub -- exercises the real transforms, the real validation, the real
# vote. Only the model is fake.
# ---------------------------------------------------------------------------

class _StubClient:
    provider = model_name = 'stub'

    def __init__(self, golds: Dict[str, float]):
        self.golds = golds
        self.calls = 0

    def _gold_for(self, text: str) -> float:
        # renderings move sentences around, so match on the rarest thing that
        # survives every transform: the numeric multiset.
        want = R.numbers(text)
        for t, g in self.golds.items():
            if R.numbers(t) == want:
                return g
        return 0.0

    def call_model(self, msgs, **kw):
        self.calls += 1
        content = msgs[-1]['content']
        if '<restated>' in content:
            body = content.split('Problem:', 1)[-1].strip()
            return f'<restated>\n{body}\n</restated>'
        return f'Answer: {self._gold_for(content)}'


def build(preset: str, stub_golds, use_sampling: bool):
    if stub_golds is not None:
        return _StubClient(stub_golds)
    from Mas_solver import (AgentRole, UnifiedLLMClient, HETEROGENEOUS_PRESETS,
                            SOLVER_VERSION, LOCAL_HF_SAMPLING)
    cfg = HETEROGENEOUS_PRESETS[preset]
    m = cfg[AgentRole.BASELINE]
    print(f"solver_version={SOLVER_VERSION} preset={preset}")
    print(f"  solver: {m.provider}/{m.model_name}")
    if use_sampling:
        LOCAL_HF_SAMPLING.update(enabled=True)
        print(f"  sampling ENABLED for arm S: {LOCAL_HF_SAMPLING}")
    # use_cache=False is load-bearing: the cache key is (provider, model,
    # messages, temperature), so k sampled calls on one problem would all be
    # the same cache entry and arm S would silently become SC@1.
    return UnifiedLLMClient(provider=m.provider, use_cache=False,
                            model_override=m.model_name,
                            load_4bit=getattr(m, 'load_4bit', False))


def solve(client, text: str, max_tokens: int,
          temperature: float = 0.0) -> Optional[float]:
    """One CoT call through the shipped baseline, so the control, the
    renderings and the samples are parsed by identical code."""
    import baselines
    try:
        return baselines.chain_of_thought(client, text, max_tokens=max_tokens,
                                          temperature=temperature).answer
    except Exception as exc:
        print(f"    solve raised {type(exc).__name__}: {str(exc)[:100]}")
        return None


def probe_sampling(client, max_tokens: int) -> bool:
    """Three sampled calls on one prompt. If they come back identical the
    sampling flag did not take, and arm S would spend four GPU hours
    re-measuring the control."""
    print("  probing sampling (3 calls)...", flush=True)
    p = ("Solve this math problem step by step. After your reasoning, state "
         "the final numeric answer on a line starting with 'Answer:'.\n\n"
         "Problem: A baker sold 17 loaves on Monday and 24 on Tuesday, then "
         "baked 31 more. How many loaves changed hands in total?")
    outs = []
    for _ in range(3):
        try:
            outs.append(str(client.call_model([{'role': 'user', 'content': p}],
                                              temperature=0.8,
                                              max_tokens=max_tokens))[:400])
        except Exception as exc:
            print(f"  SAMPLING PROBE RAISED {type(exc).__name__}: {exc}")
            return False
    uniq = len(set(outs))
    print(f"  sampling probe: {uniq}/3 distinct continuations")
    return uniq > 1


# ---------------------------------------------------------------------------

def render_all(client, text: str, names, max_tokens: int) -> List[R.Rendering]:
    """Build the rendering set. Structural ones are free; `paraphrase` is the
    only one that costs a call, and it is refused unless it is faithful."""
    out = R.build_structural(text, [n for n in names if n in R.STRUCTURAL])
    if 'paraphrase' in names:
        try:
            raw = client.call_model(
                [{'role': 'user', 'content': R.paraphrase_prompt(text)}],
                temperature=0.0, max_tokens=max_tokens)
            out.append(R.parse_paraphrase(raw, text))
        except Exception as exc:
            out.append(R.Rendering('paraphrase', text, ok=False,
                                   refused=f'{type(exc).__name__}: {exc}'))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default=MANIFEST)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--preset', default='qwen_math7b_mixed')
    ap.add_argument('--arms', default='CFS', help='any of C, F, S')
    ap.add_argument('--renderings', default=','.join(R.DEFAULT_SET))
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--max-tokens', type=int, default=1024)
    ap.add_argument('--sc-temperature', type=float, default=0.8)
    ap.add_argument('--stub', action='store_true')
    ap.add_argument('--no-resume', dest='resume', action='store_false',
                    help='re-run rows that are already complete in --out')
    ap.add_argument('--max-hours', type=float, default=0.0,
                    help="stop cleanly after this many hours (0 = no limit). "
                         "Kaggle kills a batch GPU session at 9h WITHOUT "
                         "saving; set 8.0 and re-run to resume.")
    ap.add_argument('--build-manifest', action='store_true')
    ap.add_argument('--dataset', default='gsm-symbolic-p2')
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--seed', type=int, default=44)
    args = ap.parse_args()

    if args.build_manifest:
        return build_manifest(args.dataset, args.n, args.seed, args.manifest)

    arms = [a for a in 'CFS' if a in args.arms.upper()]
    names = [s.strip() for s in args.renderings.split(',') if s.strip()]
    if 'F' in arms and 'C' not in arms:
        print("arm F reuses the control's answer as the identity vote; "
              "run it as --arms CF")
        return 2

    with open(args.manifest, encoding='utf-8') as fh:
        man = json.load(fh)
    rows: List[Dict] = man['rows'][:args.limit] if args.limit else man['rows']
    # arm S matches F's SOLVER calls, so the two arms cost the same.
    k_sc = max(2, len(names))
    print(f"rows: {len(rows)}  dataset={man.get('dataset')} seed={man.get('seed')}")
    print(f"arms: {' '.join(arms)}   renderings: {', '.join(names)}   SC k={k_sc}")

    client = build(args.preset,
                   {r['text']: r['gold'] for r in rows} if args.stub else None,
                   use_sampling=('S' in arms and not args.stub))

    if 'S' in arms and not args.stub and not probe_sampling(client, 256):
        print("\nABORT: sampling did not take, so arm S would be a second copy "
              "of the control. Fix LOCAL_HF_SAMPLING before spending the run.")
        return 3

    # merge with an earlier session's file so the arms stay paired
    prior: Dict[str, Dict] = {}
    if os.path.exists(args.out):
        try:
            with open(args.out, encoding='utf-8') as fh:
                prior = {r['pid']: r for r in json.load(fh).get('rows', [])}
            print(f"merging into {len(prior)} rows from a previous session")
            if args.resume:
                print("  resume is ON: rows already carrying every requested "
                      "arm are skipped (--no-resume to force)")
        except Exception as exc:
            print(f"could not read {args.out} ({exc}); starting fresh")

    out, t0, done, stopped_early = [], time.time(), 0, 0
    for i, r in enumerate(rows, 1):
        gold, text = r['gold'], r['text']
        rec = dict(prior.get(r['problem_id'], {}))
        rec.update(pid=r['problem_id'], gold=gold, dataset=r.get('dataset'))

        # Resume. A Kaggle session that dies at hour eight must not restart at
        # row one: this run is ~9 GPU-hours and this project has lost long runs
        # to exactly that. A row already carrying every requested arm is kept
        # and skipped, so re-running the same command continues it.
        if args.resume and all(f'{a}_correct' in rec for a in arms):
            out.append(rec)
            done += 1
            continue

        # Wall clock. Kaggle's batch GPU session ("Save & Run All") is capped
        # at 9 hours and is killed at the cap with no chance to write anything;
        # the interactive session is capped at 12. Stopping ourselves one hour
        # short turns a lost run into a resumable one, because every row so far
        # is already on disk.
        if args.max_hours and (time.time() - t0) > args.max_hours * 3600:
            stopped_early = len(rows) - i + 1
            print(f"\n  --max-hours {args.max_hours} reached with "
                  f"{stopped_early} row(s) left. The file is complete up to "
                  f"here; re-run the identical command to resume.")
            break

        if 'C' in arms:
            c = solve(client, text, args.max_tokens)
            rec.update(C_answer=c, C_correct=correct(c, gold))

        if 'F' in arms:
            rs = render_all(client, text, names, args.max_tokens)
            votes, detail = [], []
            for ren in rs:
                if ren.name == 'identity':
                    v = rec.get('C_answer')            # already paid for
                elif not ren.ok:
                    v = None                           # refused: does not vote
                else:
                    v = solve(client, ren.text, args.max_tokens)
                votes.append((ren.name, v))
                detail.append({'name': ren.name, 'ok': ren.ok,
                               'refused': ren.refused, 'answer': v})
            ans, info = R.majority(votes)
            rec.update(F_answer=ans, F_correct=correct(ans, gold),
                       F_renderings=detail, F_vote=info,
                       F_disagreement=R.disagreement_rate([v for _, v in votes]),
                       F_refused=[d['name'] for d in detail if not d['ok']])

        if 'S' in arms:
            svotes = [(f'sample_{j}', solve(client, text, args.max_tokens,
                                            temperature=args.sc_temperature))
                      for j in range(k_sc)]
            ans, info = R.majority(svotes, default_name='sample_0')
            rec.update(S_answer=ans, S_correct=correct(ans, gold),
                       S_samples=[v for _, v in svotes], S_vote=info,
                       S_disagreement=R.disagreement_rate([v for _, v in svotes]))

        out.append(rec)
        el, ran = time.time() - t0, len(out) - done
        flags = ' '.join(f"{a}={'OK ' if rec.get(f'{a}_correct') else 'BAD'}"
                         for a in arms)
        print(f"[{i:3d}/{len(rows)}] {str(r['problem_id'])[:22]:22s} {flags}"
              f"  {el/max(ran,1):5.0f}s/row"
              f"  eta {el/max(ran,1)*(len(rows)-i)/60:4.0f} min",
              flush=True)
        with open(args.out, 'w', encoding='utf-8') as fh:
            json.dump(dict(manifest=args.manifest, preset=args.preset,
                           arms=sorted(set(arms) | {k[0] for p in prior.values()
                                                    for k in p if k.endswith('_correct')}),
                           renderings=names, sc_k=k_sc,
                           sc_temperature=args.sc_temperature,
                           n=len(rows), rows=out), fh,
                      ensure_ascii=False, indent=1)

    if stopped_early:
        print(f"\n  PARTIAL: {len(out)}/{len(rows)} rows. No verdict is read "
              f"off a partial run.")
    return summarise(out, arms, names, args)


def summarise(out: List[Dict], arms, names, args) -> int:
    n = len(out)
    have = [a for a in 'CFS' if any(f'{a}_correct' in r for r in out)]
    score = {a: sum(1 for r in out if r.get(f'{a}_correct')) for a in have}
    label = {'C': 'control, greedy CoT   ', 'F': 'form-diverse majority ',
             'S': 'self-consistency      '}
    print('\n' + '=' * 72)
    for a in have:
        print(f"  arm {a}  {label[a]} {score[a]:3d}/{n} = {100*score[a]/n:5.1f}%")

    def paired(x, y):
        w = sum(1 for r in out if r.get(f'{x}_correct') and not r.get(f'{y}_correct'))
        l = sum(1 for r in out if r.get(f'{y}_correct') and not r.get(f'{x}_correct'))
        return w, l

    print()
    for x, y in (('F', 'C'), ('S', 'C'), ('F', 'S')):
        if x in have and y in have:
            w, l = paired(x, y)
            print(f"  {x} vs {y}:  W={w} L={l}  net={w-l:+d}   "
                  f"(discordant {w+l})")

    def mean_dis(key):
        vals = [r[key] for r in out if r.get(key) is not None]
        return statistics.mean(vals) if vals else None

    df, ds = mean_dis('F_disagreement'), mean_dis('S_disagreement')
    print()
    if df is not None:
        print(f"  mean disagreement among renderings : {df:.3f}")
    if ds is not None:
        print(f"  mean disagreement among samples    : {ds:.3f}")

    if 'F' in have:
        ref = [x for r in out for x in r.get('F_refused', [])]
        print(f"  renderings refused as unfaithful   : {len(ref)}"
              + (f"  { {k: ref.count(k) for k in set(ref)} }" if ref else ''))
        ties = sum(1 for r in out if (r.get('F_vote') or {}).get('tie'))
        una = sum(1 for r in out if (r.get('F_vote') or {}).get('unanimous'))
        print(f"  unanimous {una}/{n}, ties broken to identity {ties}/{n}")

    print('\n  pre-registered reading:')
    if n != 100:
        print(f"    n={n}, not the pre-registered 100 -- report as measured, no verdict")
    elif df is not None and ds is not None and df <= ds:
        print("    KILL: re-rendering does not decorrelate more than sampling")
    elif 'F' in have and 'C' in have:
        w, l = paired('F', 'C')
        if score['F'] - score['C'] >= 6 and w >= 2 * l:
            if 'S' in have:
                ws, ls = paired('F', 'S')
                print("    F beats C" + ("  and beats S -> FULL RUN"
                                          if ws >= ls + 4 else
                                          "  but not S: the gain is the extra "
                                          "calls, not the re-rendering"))
            else:
                print("    F beats C -> run arm S before concluding anything")
        elif score['F'] <= score['C']:
            print("    F <= C: input-space diversity does not pay -> direction closes")
        else:
            print("    inconclusive at n=100; report as measured")
    print(f"\n  saved: {args.out}   (download it -- Kaggle loses the session)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
