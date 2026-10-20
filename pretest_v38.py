"""
[v38.0] Symbolic Surgery (trace_surgery.py): the fresh confirmation run.

WHY
---
ADAPT-R (v35/v36) drops every sample whose own chain of thought holds a
provably false arithmetic claim. On the 680 stored rows (replay_v38.py, no GPU)
repairing those samples instead of dropping them, with the correction carried
forward by numeric matching, gives ADAPT-RS 573 vs ADAPT-R 567 (W8 L2), +4 of
it on v36's clean GSM-Hard rows, and repair-then-vote beats SC@5 W17 L0. All of
that is offline and partly in-sample; it never re-scored a repaired trace and
it could not try RESUME (continue the solver from the corrected prefix). This
run measures both, on rows nobody has looked at.

ROWS (pretest_data/v38_surgery.json, seed 38, built once from metadata only)
---------------------------------------------------------------------------
  A  200  GSM-Hard, CLEAN (gsmhard_audit.py: auditable, none of the three
          defect channels), gold > 0, max number in the text >= 1e4, NOT in
          v36's manifest nor the v18/v19 pretests. v36 showed every surgery
          opportunity lives in this pool (56 of its 61 refuted samples, all 5
          wins); its GSM-Plus pools had none. 1536 new tokens.
  M   60  MATH-500 problems with a plain numeric answer, any level, NOT in
          v34/v35 (the non-inferiority check where the audit is rarely
          triggered and propagation has hurt). 2048 new tokens.
Interleaved (M evenly spaced), so either half of a two-machine run mixes both.

SAMPLES AND THE SURGERY STAGE (the GPU part does not depend on the policy)
--------------------------------------------------------------------------
Exactly v36 per row: 5 samples from Qwen2.5-Math-7B-Instruct (v22's CoT
prompt, T 0.8, top-p 0.95, top-k 50), each scored by Qwen2.5-Math-PRM-7B.
Then, for EVERY sample the frozen audit (trace_siv.py) refutes and in which
trace_surgery.first_false finds a correctable claim:
  prop    the propagated repair, re-scored by the PRM          (no call)
  resume  ONE continuation of the corrected prefix with the same sampling
          settings and the row's token budget, scored by the PRM (one call)
Both are stored; the policy (trace_surgery.surgery_policy) is applied at
reading time and each resume it chooses is charged one call. The run refuses
to start while the policy is unwritten (pre-registration, V38_PLAN.md).

PRE-REGISTERED (V38_PLAN.md section 4; frozen before any sample exists)
----------------------------------------------------------------------
Every reading uses the canonical grader |pred - gold| < 1e-3.
  PRIMARY    ADAPT-RS vs ADAPT-R on the 200 A rows: net >= +3 with W >= 2L ->
             SUPPORTED ("significant" if sign p < 0.05); net <= 0 -> REFUTED;
             otherwise INCONCLUSIVE
  SECONDARY  ADAPT-RS vs ADAPT-R on the 60 M rows: net >= -1 -> NON-INFERIOR
Reported, not decisive: ADAPT-RS vs SC@5, the fixed policies, coverage (rows
with no right sample that gain one), the resume diagnostics, calls/row.

MODES
-----
    python pretest_v38.py --build-manifest     # once (needs `datasets`); then commit
    python pretest_v38.py --max-hours 8.5      # Kaggle T4 x2; re-run to resume
    python pretest_v38.py --reverse ...        # the second machine, from the end
    python pretest_v38.py --summary-only --out A.json --merge B.json
    python pretest_v38.py --stub --limit 6     # offline plumbing (no policy needed)
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import random
import re
import sys
import time
from collections import Counter
from typing import Dict, List, Optional

import pretest_v21 as V21
import pretest_v24 as V24
import pretest_v34 as V34
import pretest_v35 as V35
import pretest_v36 as V36
import replay_v35 as R
import score_prm_v22 as P
import trace_siv as T
import trace_surgery as S

MANIFEST = 'pretest_data/v38_surgery.json'
OUT = 'pretest_v38.json'
OUT_STUB = 'pretest_v38_stub.json'
SEED = 38
N_A, N_M = 200, 60
K = 5
TEMPERATURE = 0.8
MAX_TOKENS = {'A': 1536, 'M': 2048}
USED_A = (V36.MANIFEST,) + V36.USED_BEFORE
USED_M = (V34.MANIFEST, V35.MANIFEST)

# pre-registered bars (V38_PLAN.md section 4)
PRI_MIN_NET, PRI_RATIO, PRI_ALPHA = 3, 2, 0.05
SEC_MIN_NET = -1


def canonical(pred, gold) -> bool:
    """The project's grader, Mas_solver.check_correctness: |pred - gold| < 1e-3."""
    return pred is not None and gold is not None and abs(pred - gold) < 1e-3


# ---------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------

def _ids(paths) -> set:
    out = set()
    for f in paths:
        if os.path.exists(f):
            out |= {r['problem_id'] for r in V24._read(f)['rows']}
    return out


def interleave(a: List[Dict], m: List[Dict]) -> List[Dict]:
    """M rows evenly spaced among the A rows, one near each end."""
    n = len(a) + len(m)
    ia, im, out = iter(a), iter(m), []
    for i in range(n):
        take_m = len(m) and (i + 1) * len(m) // n > i * len(m) // n
        x = next(im if take_m else ia, None)
        out.append(x if x is not None else next(ia if take_m else im))
    return out


def build_manifest(force: bool = False) -> int:
    if os.path.exists(MANIFEST) and not force:
        print(f"{MANIFEST} exists; refusing to overwrite pre-registered rows (--force to redo)")
        return 1
    from datasets import load_dataset
    import gsmhard_audit as GA
    _, sets = GA.run()
    bad = set(sets['collision']) | set(sets['never_perturbed']) | set(sets['platinum_dropped'])
    used = _ids(USED_A)
    a = []
    for i, r in enumerate(load_dataset('reasoning-machines/gsm-hard', split='train')):
        pid = f'gsm-hard_{i}'
        g = V36._num(r['target'])
        if (pid in sets['auditable'] and pid not in bad and pid not in used and g is not None and g > 0
                and V36.max_number(r['input']) >= V36.MIN_MAG_A):
            a.append({'problem_id': pid, 'text': r['input'].strip(), 'gold': g, 'level': 'A',
                      'subject': 'gsm-hard-clean'})
    used = _ids(USED_M)
    m = []
    for i, x in enumerate(load_dataset('HuggingFaceH4/MATH-500', split='test')):
        g = V34._to_num(x['answer'].strip())
        pid = f'math500_{i}'
        if g is None or not re.fullmatch(V34._NUM, x['answer'].strip().replace(' ', '')) or pid in used:
            continue
        m.append({'problem_id': pid, 'text': x['problem'].strip(), 'gold': g, 'level': 'M',
                  'subject': f"MATH {x['subject']} L{x['level']}"})
    rng = random.Random(SEED)
    rng.shuffle(a)
    rng.shuffle(m)
    print(f"  A: {len(a)} eligible -> {min(N_A, len(a))};  M: {len(m)} eligible -> {min(N_M, len(m))}")
    rows = interleave(a[:N_A], m[:N_M])
    with open(MANIFEST, 'w', encoding='utf-8') as fh:
        json.dump({'dataset': 'v38 Symbolic Surgery: GSM-Hard clean >= 1e4 (A) + MATH-500 numeric (M), fresh',
                   'seed': SEED, 'n': len(rows), 'eligible': {'A': len(a), 'M': len(m)}, 'rows': rows},
                  fh, ensure_ascii=False, indent=1)
    print(f"wrote {len(rows)} rows -> {MANIFEST}")
    return 0


def load_rows() -> List[Dict]:
    return [{'key': r['problem_id'], 'pid': r['problem_id'], 'set': 'main', 'gold': r['gold'],
             'text': r['text'], 'level': r['level'], 'subject': r['subject']}
            for r in V24._read(MANIFEST)['rows']]


def complete(rec: Dict) -> bool:
    """5 PRM-scored samples, and the surgery stage done for each of them."""
    c = (rec or {}).get('C') or []
    return V34.complete({'C': c}) and all('surgery' in s for s in c[:K])


# ---------------------------------------------------------------------------
# the policy guard (pre-registration lives in the .py, not in a notebook cell)
# ---------------------------------------------------------------------------

def policy_written() -> bool:
    probe = {'answer': 1.0, 'prm': [0.9], 'audit': [[('1 + 1 = 3', T.FALSE)]],
             'surgery': {'site': {'claim': '1 + 1 = 3', 'claimed': 3.0, 'value': 2.0, 'step': 0,
                                  'rhs_literal': True, 'lhs': '1 + 1', 'cut': 0},
                         'slips': [(3.0, 2.0)], 'prop': {'reached': True, 'answer': 2.0}}}
    try:
        return S.surgery_policy(probe, 1) in S.ACTIONS
    except NotImplementedError:
        return False


def frozen() -> str:
    def sha(path):
        return hashlib.sha256(open(path, 'rb').read()).hexdigest()[:12]
    try:
        pol = hashlib.sha256(inspect.getsource(S.surgery_policy).encode()).hexdigest()[:12]
    except (OSError, TypeError):
        pol = 'NOT the committed function (patched at run time)'
    return (f"trace_siv.py {sha(T.__file__)}; trace_surgery.py {sha(S.__file__)}; "
            f"surgery_policy() {pol}")


# ---------------------------------------------------------------------------
# agents
# ---------------------------------------------------------------------------

class _StubSolver(V34._StubSolver):
    """Offline stand-in that writes a checkable sum, with a carry slip in a
    third of the samples (so the surgery stage has something to do)."""

    def call_model(self, msgs, temperature=0.0, max_tokens=0, **kw):
        content = msgs[-1]['content']
        g = next((v for t, v in self.gold.items() if t[:60] in content), 0.0)
        a = int(g) - 7
        slip = self.rng.random() < 0.34
        c = a + 7 + (10 if slip else 0)
        return (f"Add the parts:\n\\[ {a} + 7 = {c} \\]\n\nKeep it:\n\\[ {c} \\times 1 = {c} \\]\n\n"
                f"So the answer is $\\boxed{{{c}}}$.")


MIN_RESUME_TOKENS = 128


def remaining_budget(client, prefix: str, max_tokens: int) -> int:
    """New tokens left for a continuation: the row's budget minus the prefix, so a
    resumed trace is never longer than a sample could have been (and stays well
    inside Qwen2.5-Math's 4096-token context), but at least MIN_RESUME_TOKENS."""
    try:
        client._ensure_local_model()
        used = len(client._local_tokenizer(prefix, add_special_tokens=False)['input_ids'])
    except Exception:
        return max_tokens
    return max(MIN_RESUME_TOKENS, max_tokens - used)


def resume_call(client, content: str, prefix: str, temperature: float, max_tokens: int) -> str:
    """ONE continuation of an assistant turn that already begins with `prefix`
    (same chat template and sampling knobs as the samples, the remainder of the
    row's token budget). Full text back."""
    if isinstance(client, _StubSolver):
        tail = re.findall(r'\d+(?:\.\d+)?', prefix[-40:])
        v = tail[-1] if tail else '0'
        return prefix + f" \\]\n\nKeep it:\n\\[ {v} \\times 1 = {v} \\]\n\nSo the answer is $\\boxed{{{v}}}$."
    import pretest_v22 as V22
    return V22.continue_from(client, content, prefix, 1, temperature, remaining_budget(client, prefix, max_tokens))[0]


MAX_ERRORS = 3          # more surgery-stage errors than this in one run is systematic: stop
_ERRORS: List[str] = []


def _error(where: str, exc: Exception) -> str:
    msg = f"{where}: {type(exc).__name__}: {str(exc)[:200]}"
    _ERRORS.append(msg)
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    print(f"  SURGERY-STAGE ERROR ({len(_ERRORS)}/{MAX_ERRORS} tolerated): {msg}", flush=True)
    if len(_ERRORS) > MAX_ERRORS:
        raise RuntimeError(f"{len(_ERRORS)} surgery-stage errors: systematic, stopping (the file is saved; "
                           f"fix and re-run the identical command to resume)") from exc
    return msg


def operate_sample(s: Dict, text: str, client, scorer, max_tokens: int) -> None:
    """The surgery stage of one sample: s['surgery'] = None (not refuted, or no
    correctable claim) or {'site', 'slips', 'prop', 'prefix', 'resume'}. A
    failing audit or resume is recorded ('surgery_error', 'resume_error'), the
    run goes on, and more than MAX_ERRORS of them stop it."""
    raw = s.get('raw') or ''
    try:
        if s.get('answer') is None or not T.refutes(T.audit(raw), s.get('prm')):
            s['surgery'] = None
            return
        op = S.operate(raw, s.get('answer'))
    except Exception as exc:                                 # the model-free part: never lose the row to it
        s['surgery'], s['surgery_error'] = None, _error('operate', exc)
        return
    if op is None:
        s['surgery'] = None
        return
    op['prop']['prm'] = scorer.score(text, P.split_steps(op['prop']['raw']))
    op['resume'] = None
    if op.get('prefix'):
        try:
            full = resume_call(client, V21.COT_PROMPT.format(problem=text), op['prefix'], TEMPERATURE, max_tokens)
            op['resume'] = {'raw': str(full), 'answer': V34.parse_answer(full),
                            'prm': scorer.score(text, P.split_steps(full))}
        except Exception as exc:                             # CUDA OOM included; _error stops a systematic failure
            op['resume_error'] = _error('resume', exc)
    s['surgery'] = op


def smoke_resume(client) -> None:
    """Fail in a minute, not after hours: one short continuation of a fixed
    corrected prefix must come back as prefix + new text."""
    prefix = "We add the parts.\n\\[ 61 + 8 + 13 + 10 + 58 = 150"
    full = resume_call(client, V21.COT_PROMPT.format(problem='What is 61 + 8 + 13 + 10 + 58?'), prefix,
                       TEMPERATURE, 24)
    if not (isinstance(full, str) and full.startswith(prefix) and len(full) > len(prefix)):
        raise RuntimeError(f"resume smoke test failed: the continuation does not extend the prefix: {full!r:.300}")
    print(f"  resume smoke test OK: ...{full[len(prefix) - 12:][:80]!r}", flush=True)


# ---------------------------------------------------------------------------
# reading the results
# ---------------------------------------------------------------------------

def _prepare(rows: List[Dict], recs: Dict) -> List[Dict]:
    """Complete rows as replay rows; audits recomputed with the frozen checker."""
    done = []
    for r in rows:
        rec = recs.get(r['key']) or {}
        if not complete(rec):
            continue
        c = []
        for x in rec['C'][:K]:
            y = dict(x)
            y['ok'] = canonical(y.get('answer'), r['gold'])
            y['audit'] = T.audit(y.get('raw') or '')
            op = x.get('surgery')
            if op:
                y['surgery'] = dict(op)
                for kind in ('prop', 'resume'):
                    if op.get(kind):
                        z = dict(op[kind])
                        z['audit'] = T.audit(z.get('raw') or '')
                        z['ok'] = canonical(z.get('answer'), r['gold'])
                        y['surgery'][kind] = z
            c.append(y)
        done.append({'set': 'main', 'pid': r['pid'], 'gold': r['gold'], 'pool': r['level'], 'C': c})
    return done


def _install_arms(policy_ok: bool) -> List[str]:
    reach = lambda x, n: S.PROPAGATE if x['surgery']['prop']['reached'] else S.REFUTE      # noqa: E731
    R.ARMS['ADAPT-RS:reach'] = lambda c: S.adapt_rs(c, reach)
    R.ARMS['ADAPT-RS:resume'] = lambda c: S.adapt_rs(c, lambda x, n: S.RESUME)
    R.ARMS['ADAPT-RS:prop'] = lambda c: S.adapt_rs(c, lambda x, n: S.PROPAGATE)
    arms = ['SC5', 'SC5-R', 'B5', 'ADAPT', 'ADAPT-R', 'ADAPT-RS:reach', 'ADAPT-RS:resume', 'ADAPT-RS:prop']
    if policy_ok:
        R.ARMS['ADAPT-RS'] = lambda c: S.adapt_rs(c)
        arms.append('ADAPT-RS')
    return arms


def _wl(rows, x, y):
    w = sum(R.right(r, x) and not R.right(r, y) for r in rows)
    l = sum(R.right(r, y) and not R.right(r, x) for r in rows)
    return w, l


def verdicts(a_rows: List[Dict], m_rows: List[Dict]) -> List[str]:
    out = []
    w, l = _wl(a_rows, 'ADAPT-RS', 'ADAPT-R')
    facts = R.wl(a_rows, 'ADAPT-RS', 'ADAPT-R')
    if w - l >= PRI_MIN_NET and w >= PRI_RATIO * l:
        sig = ' and significant' if R.sign_p(w, l) < PRI_ALPHA else ''
        out.append(f"PRIMARY    SUPPORTED{sig}: surgery improves refutation on arithmetic-heavy rows ({facts})")
    elif w - l <= 0:
        out.append(f"PRIMARY    REFUTED: {facts}")
    else:
        out.append(f"PRIMARY    INCONCLUSIVE: {facts}")
    w, l = _wl(m_rows, 'ADAPT-RS', 'ADAPT-R')
    out.append(f"SECONDARY  {'NON-INFERIOR' if w - l >= SEC_MIN_NET else 'NOT MET'} on MATH-500: "
               f"{R.wl(m_rows, 'ADAPT-RS', 'ADAPT-R')}")
    return out


def diagnostics(done: List[Dict]) -> None:
    ops = [(r, x) for r in done for x in r['C'] if x.get('surgery')]
    refuted = sum(T.refutes(x['audit'], None) for r in done for x in r['C'] if x.get('answer') is not None)
    print(f"\n  surgery stage: {refuted} refuted samples, {len(ops)} with a correctable claim")
    for kind in ('prop', 'resume'):
        rep = [(r, x, x['surgery'][kind]) for r, x in ops if x['surgery'].get(kind)]
        if not rep:
            continue
        wrong = [(r, x, z) for r, x, z in rep if not x['ok']]
        right = [(r, x, z) for r, x, z in rep if x['ok']]
        still = sum(T.refutes(z['audit'], None) for _, _, z in rep)
        same = sum(z.get('answer') is not None and x.get('answer') is not None
                   and abs(z['answer'] - x['answer']) < 1e-9 for _, x, z in rep)
        print(f"    {kind:6s} n={len(rep):3d}  rescued {sum(z['ok'] for _, _, z in wrong)}/{len(wrong)} wrong, "
              f"broke {sum(not z['ok'] for _, _, z in right)}/{len(right)} right; re-audit still refutes {still}; "
              f"answer unchanged {same}; mean last-step PRM {sum(z['prm'][-1] for _, _, z in rep) / len(rep):.3f}")
    corr = [(r, x) for r, x in ops if S.n_sharing(r['C'], r['C'].index(x), range(K)) >= 2]
    print(f"    correlated slips (first false claim shared by >= 2 of the 5 samples): {len(corr)} samples")
    errs = sum(bool(x.get('surgery_error')) or bool((x.get('surgery') or {}).get('resume_error'))
               for r in done for x in r['C'])
    print(f"    surgery-stage errors recorded during the run: {errs}" + ('  <-- READ THEM' if errs else ''))
    acts = Counter()
    if policy_written():
        for r in done:
            for i, x in enumerate(r['C']):
                if x.get('surgery'):
                    acts[S.surgery_policy(x, S.n_sharing(r['C'], i, range(K)))] += 1
        print(f"    the frozen policy over all 5 samples of each row: {dict(acts)}")
    zero = [r for r in done if not any(x['ok'] for x in r['C'])]
    gain = [r for r in zero if any(x.get('surgery') and any((x['surgery'].get(k) or {}).get('ok')
                                                              for k in ('prop', 'resume')) for x in r['C'])]
    print(f"    coverage: {len(gain)} of {len(zero)} rows with no right sample gain one by surgery")


def summarise(rows: List[Dict], recs: Dict) -> int:
    keep = V21.correct
    V21.correct = canonical                                  # R.right and the plurality read it at call time
    try:
        return _summarise(rows, recs)
    finally:
        V21.correct = keep


def _summarise(rows: List[Dict], recs: Dict) -> int:
    done = _prepare(rows, recs)
    print('\n' + '=' * 78)
    print(f"  v38 Symbolic Surgery (fresh): {len(done)}/{len(rows)} rows complete; canonical grader abs < 1e-3")
    print(f"  frozen: {frozen()}")
    ok = policy_written()
    if not ok:
        print("  trace_surgery.surgery_policy() is not written: only the fixed policies are read, no verdict.")
    a_rows = [r for r in done if r['pool'] == 'A']
    m_rows = [r for r in done if r['pool'] == 'M']
    if done:
        arms = _install_arms(ok)
        groups = {'A (GSM-Hard)': a_rows, 'M (MATH)': m_rows, 'ALL': done}
        print(f"\n  {'arm':16s}" + ''.join(f"{g:>14s}" for g in groups) + '   calls/row (ALL)')
        for arm in arms:
            calls = sum(R.ARMS[arm](r['C'])[1] for r in done) / len(done)
            print(f"  {arm:16s}" + ''.join(f"{sum(R.right(r, arm) for r in g):8d}/{len(g):<5d}" for g in groups.values())
                  + f"   {calls:.2f}")
        print()
        for x in arms[5:]:
            print(f"  A    {R.wl(a_rows, x, 'ADAPT-R')}")
        if ok:
            print(f"  A    {R.wl(a_rows, 'ADAPT-RS', 'SC5')}")
            print(f"  M    {R.wl(m_rows, 'ADAPT-RS', 'SC5')}")
        diagnostics(done)
        R.audit_stats(done, 'fresh')
    print('\n  pre-registered reading:')
    if len(done) < len(rows):
        print(f"    none: PARTIAL run, {len(done)}/{len(rows)} rows complete. Re-run the identical command.")
    elif not ok:
        print("    none: write and commit trace_surgery.surgery_policy(), then re-read.")
    else:
        for line in verdicts(a_rows, m_rows):
            print('    ' + line)
    return 0


# ---------------------------------------------------------------------------
# runner (v34's, plus the surgery stage)
# ---------------------------------------------------------------------------

def run(args) -> int:
    if not os.path.exists(MANIFEST):
        print(f"missing {MANIFEST}: run `python pretest_v38.py --build-manifest` first")
        return 2
    if not args.stub and not policy_written():
        print("REFUSING TO START: trace_surgery.surgery_policy() is not written. The policy is part of the "
              "pre-registration (V38_PLAN.md section 3): write it, run test_v38.py, commit, push, then run.")
        return 3
    rows = load_rows()
    if args.reverse:
        rows = list(reversed(rows))
    if args.limit:
        rows = rows[:args.limit]
    out = args.out or (OUT_STUB if args.stub else OUT)
    recs: Dict[str, Dict] = {}
    if os.path.exists(out) and args.resume:
        recs = V24._read(out).get('rows', {})
        print(f"resuming: {len(recs)} rows in {out}")
    meta = {'manifest': MANIFEST, 'k': K, 'temperature': TEMPERATURE, 'max_tokens': MAX_TOKENS,
            'prompt': V21.COT_PROMPT, 'verifier': P.PRM, 'gate_min': T.GATE_MIN, 'frozen': frozen()}
    todo = [r for r in rows if not complete(recs.get(r['key'], {}))]
    print(f"rows: {len(rows)}  to do: {len(todo)}  frozen: {meta['frozen']}", flush=True)
    if not todo:
        return summarise(load_rows(), recs)
    if not args.stub:
        try:
            import pretest_v32 as T32
            T32.scrub_git_credentials(os.path.dirname(os.path.abspath(__file__)))
        except Exception:
            pass
    if not args.stub:
        import pretest_v22  # noqa: F401  (resume_call needs it: fail now, not at the first refuted sample)
    solver = _StubSolver(rows) if args.stub else V21.build_client(args.preset, None)
    scorer = (V34._StubPRM(rows) if args.stub
              else P.PRMScorer(P.PRM, four_bit=True, device_map={'': 1 if V34._n_gpus() > 1 else 0}))
    smoke_resume(solver)        # after the PRM, so the solver loads in the v34/v36 order (PRM first)
    t0 = time.time()
    deadline = t0 + args.max_hours * 3600 if args.max_hours else 0.0
    for i, row in enumerate(todo, 1):
        if deadline and time.time() >= deadline:
            V34._save(out, recs, meta)
            print("\n  --max-hours reached; re-run the identical command to resume.")
            return summarise(load_rows(), recs)
        rec = recs.setdefault(row['key'], {})
        c = rec.setdefault('C', [])
        budget = MAX_TOKENS[row['level']]
        ts = time.time()
        if len(c) < K:
            raws = V21.sample(solver, V21.COT_PROMPT.format(problem=row['text']), K - len(c), TEMPERATURE, budget)
            c += [{'raw': str(x), 'answer': V34.parse_answer(x)} for x in raws]
        for s in c:
            if s.get('prm') is None:
                s['prm'] = scorer.score(row['text'], P.split_steps(s['raw']))
        V34._save(out, recs, meta)
        for s in c[:K]:
            if 'surgery' not in s:
                operate_sample(s, row['text'], solver, scorer, budget)
        rec['seconds'] = round(time.time() - ts, 1)
        V34._save(out, recs, meta)
        right = sum(canonical(s['answer'], row['gold']) for s in c[:K])
        ops = [s['surgery'] for s in c[:K] if s.get('surgery')]
        fixed = sum(canonical((o.get('resume') or {}).get('answer'), row['gold']) for o in ops)
        print(f"[{i:3d}/{len(todo)}] {row['pid']:16s} {row['level']} right {right}/{K}  surgery {len(ops)} "
              f"(resume right {fixed})  {rec['seconds']:5.0f}s  {(time.time() - t0) / i:5.0f}s/row", flush=True)
    V34._save(out, recs, meta)
    print(f"\n  saved: {out}")
    return summarise(load_rows(), recs)


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
    ap.add_argument('--reverse', action='store_true', help='work through the rows from the end')
    ap.add_argument('--merge', action='append', default=[], help='more result files (summary only)')
    args = ap.parse_args()
    if args.build_manifest:
        return build_manifest(args.force)
    if args.summary_only:
        path = args.out or (OUT_STUB if args.stub else OUT)
        rows = load_rows()
        if args.limit:
            rows = rows[:args.limit]
        return summarise(rows, merged([path] + args.merge))
    return run(args)


def merged(paths: List[str]) -> Dict:
    """Rows of several result files; a complete record wins over a partial one."""
    out: Dict = {}
    for p in paths:
        if not os.path.exists(p):
            print(f"  (missing: {p})")
            continue
        for k, rec in V24._read(p).get('rows', {}).items():
            if k not in out or (complete(rec) and not complete(out[k])):
                out[k] = rec
    return out


if __name__ == '__main__':
    sys.exit(main())
