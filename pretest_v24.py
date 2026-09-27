"""
[v24.0] Pre-test: the Reader agent. Does a separate Reader, one that makes
each sentence's situation explicit, improve the Solver's reasoning on
GSM-Symbolic P2 at equal solver samples?

The method is in situation_reader.py. The evidence that motivates it is in
v24_diagnosis.py (offline, no GPU). In one line: the Solver's remaining P2
errors are template-level misreadings of the situation, so resampling
re-draws them. MathWorld showed that situation QAs placed at the sentence
help a solver, but only with gold world models. v24 has a Reader agent
write them.

ARMS (k solver samples each; T=0.8, top-p 0.95, top-k 50, 1024 tokens: the
v22 settings)
----
  C     plain CoT, the prompt of every earlier version. On --dev rows it is the
        5 samples the v22 run stored; in the confirmation it is drawn here.
  ASQ   the Reader's notes interleaved after each sentence            <- v24
  END   the same notes appended after the unchanged problem
        (placement control, MathWorld's "all at once")
  SELF  one call; the Solver is told to read sentence by sentence first
        (single-agent control: PS+/DUP inside one prompt)
  ASQM  optional: ASQ with the MATH model as the Reader (heterogeneity control)
The Reader is called once per row, greedily, and its reading is stored, so
ASQ and END always carry the same notes. A reading with nothing usable
leaves ASQ/END byte-identical to C on that row, and the summary counts it.

PRE-REGISTERED, frozen before any ASQ sample exists (2026-09-27)
-----------------------------------------------------------------
Metric. Per-sample accuracy is the mean over rows of (correct samples / k):
how often one reasoning chain is right. SC@5 (plurality) is the system
metric. GSM-Symbolic rows cluster by template (68% of the row variance is
between templates, v24_diagnosis.py), so every reading also counts templates.
Dev pass (--dev): the 100 main + 20 guard rows of the v22 confirmation run,
with their stored C samples.
  PRIMARY (main)   ASQ - C per-sample >= +4.0 pp, AND a paired sign-flip
                   permutation test over rows gives p < 0.05, AND the gain
                   stays >= +2.0 pp after dropping the single template that
                   helps most                              -> SUPPORTED
                   ASQ - C <= 0                              -> REFUTED
                   otherwise                                 -> INCONCLUSIVE
                   Power, simulated on the stored C samples (v24_diagnosis.py
                   --power): about 0.7 for a +6 pp gain made of big gains on
                   the misread templates and small losses on easy rows, about
                   0.8 for +5 pp on the misread templates alone; a false
                   SUPPORTED about 1% of the time with no gain. The first
                   version of this rule used a sign test and more-templates-
                   up-than-down. It was replaced before any ASQ sample
                   existed, because the same simulation gave it 0.2 power on
                   the likely effect shape.
  SYSTEM  (main)   SC@5 ASQ vs SC@5 C: net >= +4 rows and W >= 2L -> the Reader-
                   Solver system beats SC@5 at equal solver samples
  MECHANISM        ASQ - END >= +2.0 pp -> the position matters (MathWorld's
                   interleaving effect holds for self-generated notes);
                   END - C >= +2.0 pp and |ASQ - END| < 2.0 -> the content
                   matters, not the position
  MULTI-AGENT      ASQ - SELF >= +2.0 pp -> a separate Reader beats telling the
                   Solver to read carefully
  GUARD   (guard)  SC@5 ASQ - SC@5 C >= -1 row -> no harm on distractor rows
Fresh confirmation (default mode): 100 new P2 rows (seed 47) plus 20 guard
rows, with C drawn in the same session. The same rules apply.

MODES
-----
    python pretest_v24.py --dev --max-hours 8.0             # v22 rows, ASQ, ~6.5 h
    python pretest_v24.py --dev --arms ASQ,END --max-hours 8 # adds END only, ~5.5 h
    python pretest_v24.py --dev --summary-only               # re-read, no GPU
    python pretest_v24.py --build-manifest                   # needs `datasets`; once, then commit
    python pretest_v24.py --max-hours 8.0                    # confirmation, C + ASQ (two sessions)
    python pretest_v24.py --stub --dev --limit 8             # offline plumbing, no GPU
Re-running the identical command RESUMES. A row keeps its reading and every
sample it already has, so adding an arm later draws only that arm.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import re
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import pretest_v21 as V21
import score_prm_v22 as P
import situation_reader as SR

POOLS = {
    'v22': {'results': 'results_September/preset_v22.json',
            'manifests': ('pretest_data/v22_confirm_p2.json', 'pretest_data/v22_confirm_guard.json')},
    'v21': {'traces': 'pretest_data/v21_traces.json',
            'manifests': ('pretest_data/v20_problems.json', 'pretest_data/v21_guard_distraction.json')},
}
MANIFESTS = (('main', 'pretest_data/v24_confirm_p2.json'),
             ('guard', 'pretest_data/v24_confirm_guard.json'))
# every row an earlier version drew, so the confirmation rows are new to all of them
EARLIER_MANIFESTS = ('pretest_data/v20_problems.json', 'pretest_data/v21_guard_distraction.json',
                     'pretest_data/v22_confirm_p2.json', 'pretest_data/v22_confirm_guard.json',
                     'pretest_data/v23_confirm_p2.json', 'pretest_data/v23_confirm_guard.json')
OUT = 'pretest_v24.json'
OUT_DEV = 'pretest_v24_dev.json'
OUT_STUB = 'pretest_v24_stub.json'   # a stub run never writes the real files
ARMS = ('C', 'ASQ', 'END', 'SELF', 'ASQM')
DEFAULT_DEV_ARMS = 'ASQ'
DEFAULT_CONFIRM_ARMS = 'C,ASQ'
CONFIRM_SEED = 47

# the pre-registered bars
PRIMARY_MIN_PP = 4.0
PRIMARY_ALPHA = 0.05
PRIMARY_LOTO_PP = 2.0
SYSTEM_MIN_NET = 4
MECH_MIN_PP = 2.0
GUARD_MIN_NET = -1


def _read(path: str) -> Dict:
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def _texts(paths: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in paths:
        for r in _read(p)['rows']:
            out[r['problem_id']] = r['text']
    return out


def template_of(pid: str) -> Optional[int]:
    """GSM-Symbolic keeps the 50 instances of a template together, so the row
    index // 50 is the template. None for rows of any other dataset."""
    m = re.match(r'gsm-symbolic_p2_(\d+)$', str(pid))
    return int(m.group(1)) // 50 if m else None


def _sample(raw: str, answer=None) -> Dict:
    return {'raw': str(raw)[:4000], 'answer': V21.parse_answer(raw) if answer is None else answer}


# ---------------------------------------------------------------------------
# the dev pool: stored C samples, nothing re-drawn
# ---------------------------------------------------------------------------

def load_pool(sources: Sequence[str] = ('v22',)) -> List[Dict]:
    rows: List[Dict] = []
    for src in sources:
        cfg = POOLS[src]
        texts = _texts(cfg['manifests'])
        if src == 'v22':
            stored = [(r['pid'], r['set'], r['gold'], r['samples'])
                      for r in _read(cfg['results'])['rows']]
        else:
            stored = [(r['pid'], r['set'], r['gold'], r['samples'])
                      for r in _read(cfg['traces'])['rows']]
        for pid, tag, gold, samples in stored:
            rows.append({'key': f'{src}:{pid}', 'pid': pid, 'source': src, 'set': tag,
                         'gold': gold, 'text': texts[pid], 'template': template_of(pid),
                         'stored_C': [{'raw': str(s.get('raw', ''))[:4000],
                                       'answer': s.get('answer')} for s in samples]})
    return rows


def load_confirm(limit: int = 0) -> List[Dict]:
    rows: List[Dict] = []
    for tag, path in MANIFESTS:
        rr = _read(path)['rows']
        for r in (rr[:limit] if limit else rr):
            rows.append({'key': f"v24:{r['problem_id']}", 'pid': r['problem_id'],
                         'source': 'v24', 'set': tag, 'gold': r['gold'], 'text': r['text'],
                         'template': template_of(r['problem_id'])})
    return rows


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

class _StubSolver:
    """Offline stand-in. Right with a probability that depends only on which
    arm's prompt it gets, so every code path runs and the arms differ."""
    provider = model_name = 'stub'

    def __init__(self, rows: List[Dict], seed: int = 0):
        self.rows = rows
        self.rng = random.Random(seed)
        self.calls = 0

    def _row(self, content: str) -> Dict:
        for r in self.rows:
            if SR.sentences(r['text'])[0] in content:
                return r
        return self.rows[0]

    def call_model(self, msgs, temperature=0.0, max_tokens=0, **kw):
        self.calls += 1
        content = msgs[-1]['content']
        if content.startswith('Problem sentences:'):          # asked to READ (ASQM)
            return "Let's solve it. We add everything up.\nAnswer: 3"
        r = self._row(content)
        g = float(r['gold'])
        if SR.ASQ_INTRO in content:
            p = 0.75
        elif SR.END_HEADER in content:
            p = 0.65
        elif 'sentence by sentence' in content:
            p = 0.5
        else:
            p = 0.55
        a = g if self.rng.random() < p else g + self.rng.choice([1, 2, 3])
        return f"We compute \\[ {a:g} \\]\nAnswer: {a:g}"


class _StubReader:
    """Offline Reader: one note per sentence that holds a number."""
    provider = model_name = 'stub-reader'

    def __init__(self):
        self.calls = 0

    def call_model(self, msgs, temperature=0.0, max_tokens=0, **kw):
        self.calls += 1
        out = []
        for line in msgs[-1]['content'].splitlines()[1:]:
            m = re.match(r'S(\d+): (.*)', line)
            if not m:
                continue
            nums = SR.numbers(m.group(2))
            out.append(f"S{m.group(1)}: Q: What number does this sentence give? A: {nums[0]:g}."
                       if nums else f"S{m.group(1)}: -")
        out.append("Asked: the quantity the question asks for.")
        return '\n'.join(out)


def build(args, rows):
    if args.stub:
        return _StubSolver(rows), _StubReader()
    import torch
    from Mas_solver import AgentRole, HETEROGENEOUS_PRESETS, UnifiedLLMClient
    two = torch.cuda.device_count() > 1
    solver = V21.build_client(args.preset, None)        # the v21/v22 solver, unchanged
    cfg = HETEROGENEOUS_PRESETS[args.preset]
    m = cfg[AgentRole.MATHEMATICIAN]                    # the parse-critical role's model
    if m.model_name == cfg[AgentRole.BASELINE].model_name:
        print("  reader: same model as the solver")
        return solver, solver
    print(f"  reader: {m.provider}/{m.model_name} (greedy) on cuda:{1 if two else 0}")
    reader = UnifiedLLMClient(provider=m.provider, use_cache=False, model_override=m.model_name,
                              load_4bit=getattr(m, 'load_4bit', False),
                              device_index=1 if two else None)
    return solver, reader


# ---------------------------------------------------------------------------
# one row
# ---------------------------------------------------------------------------

READER_ABORT_AFTER = 5
_READS = {'n': 0, 'ok': 0, 'shown': False}


def read(reader, text: str, guard: bool = True) -> Dict:
    """One greedy Reader call. A failed or unusable read degrades that row to
    C. The guard is there because a Reader that fails on EVERY row (the model
    will not load, or it writes a solution instead of the format) would
    otherwise burn a whole session measuring C against C: if the first
    READER_ABORT_AFTER reads of a session are all unusable, the run stops
    and shows what the Reader wrote."""
    t0 = time.time()
    err = ''
    try:
        raw = str(reader.call_model(SR.reader_messages(text), temperature=0.0,
                                    max_tokens=SR.READER_MAX_TOKENS))
    except Exception as exc:
        raw, err = '', f'{type(exc).__name__}: {exc}'[:200]
    out = SR.parse_reading(raw, len(SR.sentences(text)))
    out.update(raw=raw[:4000], error=err, seconds=round(time.time() - t0, 1))
    if guard:
        _READS['n'] += 1
        _READS['ok'] += bool(out['ok'])
        if out['ok'] and not _READS['shown']:
            _READS['shown'] = True
            print("  first reading of this session (check it looks like the demos):\n    "
                  + raw.strip().replace('\n', '\n    '), flush=True)
        if _READS['n'] >= READER_ABORT_AFTER and not _READS['ok']:
            raise RuntimeError(f"the Reader gave nothing usable on its first {_READS['n']} rows; "
                               f"last error: {err or 'none'}; last output: {raw[:600]!r}")
    return out


def prompt_for(arm: str, text: str, rec: Dict) -> str:
    if arm == 'C':
        return SR.cot_prompt(text)
    if arm == 'ASQ':
        return SR.asq_prompt(text, rec['reading'])
    if arm == 'END':
        return SR.end_prompt(text, rec['reading'])
    if arm == 'SELF':
        return SR.self_prompt(text)
    if arm == 'ASQM':
        return SR.asq_prompt(text, rec['reading_m'])
    raise ValueError(arm)


def new_record(row: Dict) -> Dict:
    return {'key': row['key'], 'pid': row['pid'], 'source': row['source'], 'set': row['set'],
            'gold': row['gold'], 'template': row['template'], 'arms': {}}


def fill(solver, reader, rec: Dict, text: str, arms: Sequence[str], args,
         deadline: float, dev: bool) -> Tuple[int, bool]:
    """Draw every requested arm this row does not have yet. Returns (arms
    drawn, stopped by the deadline)."""
    made = 0
    for arm in arms:
        have = rec['arms'].get(arm, {}).get('samples', [])
        if len(have) >= args.k or (dev and arm == 'C'):
            continue
        if deadline and time.time() > deadline:
            return made, True
        if arm in ('ASQ', 'END') and 'reading' not in rec:
            rec['reading'] = read(reader, text)
        if arm == 'ASQM' and 'reading_m' not in rec:
            rec['reading_m'] = read(solver, text, guard=False)   # its failures are the finding
        t0 = time.time()
        raws = V21.sample(solver, prompt_for(arm, text, rec), args.k - len(have),
                          args.temperature, args.max_tokens)
        rec['arms'][arm] = {'samples': have + [_sample(r) for r in raws],
                            'seconds': round(rec['arms'].get(arm, {}).get('seconds', 0)
                                             + time.time() - t0, 1)}
        made += 1
    return made, False


# ---------------------------------------------------------------------------
# reading the results
# ---------------------------------------------------------------------------

def answers(rec: Dict, arm: str, k: int) -> List[Optional[float]]:
    return [s['answer'] for s in rec['arms'].get(arm, {}).get('samples', [])[:k]]


def has(rec: Dict, arm: str, k: int) -> bool:
    return len(rec['arms'].get(arm, {}).get('samples', [])) >= k


def acc(rec: Dict, arm: str, k: int) -> float:
    a = answers(rec, arm, k)
    return sum(V21.correct(x, rec['gold']) for x in a) / len(a) if a else 0.0


def sc(rec: Dict, arm: str, k: int) -> bool:
    return V21.correct(P.plain_vote(answers(rec, arm, k)), rec['gold'])


def first(rec: Dict, arm: str) -> bool:
    a = answers(rec, arm, 1)
    return bool(a) and V21.correct(a[0], rec['gold'])


def oracle(rec: Dict, arm: str, k: int) -> bool:
    return any(V21.correct(x, rec['gold']) for x in answers(rec, arm, k))


def sign_p(w: int, l: int) -> float:
    """Exact two-sided sign test (reported; the verdict uses perm_p)."""
    n = w + l
    if not n:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def perm_p(diffs: Sequence[float], b: int = 10000, seed: int = 0) -> float:
    """Two-sided paired sign-flip permutation test on the per-row differences.
    It uses the magnitudes: a row that goes from 0/5 to 4/5 weighs four times a
    row that loses one sample. The sign test cannot tell those apart, and in
    power simulations on the stored C samples that made it miss the likely
    effect shape (big gains on a few misread templates, small losses on easy
    rows). Seeded, so a summary always prints the same p."""
    d = [v for v in diffs if abs(v) > 1e-12]
    if not d:
        return 1.0
    obs = abs(sum(d)) - 1e-9
    rng = random.Random(seed)
    hit = sum(1 for _ in range(b) if abs(sum(v if rng.random() < 0.5 else -v for v in d)) >= obs)
    return (hit + 1) / (b + 1)


def _cluster(rec: Dict):
    return rec['template'] if rec['template'] is not None else rec['key']


def compare(rows: List[Dict], x: str, y: str, k: int) -> Dict:
    """Paired per-sample comparison of arm x against arm y."""
    d = [acc(r, x, k) - acc(r, y, k) for r in rows]
    w = sum(1 for v in d if v > 1e-9)
    l = sum(1 for v in d if v < -1e-9)
    by = collections.defaultdict(list)
    for r, v in zip(rows, d):
        by[_cluster(r)].append(v)
    tw = sum(1 for v in by.values() if sum(v) / len(v) > 1e-9)
    tl = sum(1 for v in by.values() if sum(v) / len(v) < -1e-9)
    # the gain left after dropping the single template that helps most: a
    # verdict must not rest on one misread template being fixed
    tot = sum(d)
    loto = min((100 * (tot - sum(v)) / (len(d) - len(v)) for v in by.values() if len(d) > len(v)),
               default=100 * tot / len(d) if d else 0.0)
    sw = sum(1 for r in rows if sc(r, x, k) and not sc(r, y, k))
    sl = sum(1 for r in rows if sc(r, y, k) and not sc(r, x, k))
    return {'pp': 100 * tot / len(d) if d else 0.0, 'w': w, 'l': l, 'p': perm_p(d),
            'sign_p': sign_p(w, l), 'tw': tw, 'tl': tl, 'n_t': len(by), 'loto': loto,
            'sw': sw, 'sl': sl, 'ci': cluster_ci(rows, x, y, k)}


def cluster_ci(rows: List[Dict], x: str, y: str, k: int, b: int = 2000,
               seed: int = 0) -> Tuple[float, float]:
    """95% bootstrap interval of the per-sample difference, resampling
    templates (the unit that errors cluster in), not rows."""
    by = collections.defaultdict(list)
    for r in rows:
        by[_cluster(r)].append(acc(r, x, k) - acc(r, y, k))
    keys = list(by)
    if not keys:
        return 0.0, 0.0
    rng = random.Random(seed)
    stats = []
    for _ in range(b):
        vals = [v for key in (rng.choice(keys) for _ in keys) for v in by[key]]
        stats.append(100 * sum(vals) / len(vals))
    stats.sort()
    return stats[int(0.025 * b)], stats[int(0.975 * b) - 1]


def _reader_line(rows: List[Dict], field: str = 'reading') -> str:
    rr = [r[field] for r in rows if field in r]
    if not rr:
        return ''
    ok = sum(1 for x in rr if x.get('ok'))
    notes = sum(x.get('n_notes', 0) for x in rr) / len(rr)
    gold = sum(1 for r in rows if field in r and SR.states_gold(r[field], r['gold']))
    secs = sum(x.get('seconds', 0) for x in rr) / len(rr)
    return (f"    reader ({field}): usable {ok}/{len(rr)}, {notes:.1f} notes/row, "
            f"{secs:.0f} s/row; a note states the gold on {gold} rows (post hoc)")


def block(rows: List[Dict], arms: Sequence[str], k: int, title: str) -> None:
    n = len(rows)
    if not n:
        return
    shown = [a for a in ARMS if a in arms or a == 'C']
    print('\n' + '=' * 76)
    print(f"  {title}  n={n}  templates={len({_cluster(r) for r in rows})}")
    for f in ('reading', 'reading_m'):
        line = _reader_line(rows, f)
        if line:
            print(line)
    print(f"    {'arm':5s} {'per-sample':>10s} {'first':>6s} {'SC@3':>5s} {f'SC@{k}':>5s} "
          f"{f'oracle@{k}':>9s}")
    for a in shown:
        if not all(has(r, a, k) for r in rows):
            continue
        ps = 100 * sum(acc(r, a, k) for r in rows) / n
        print(f"    {a:5s} {ps:9.1f}% {sum(first(r, a) for r in rows):6d} "
              f"{sum(sc(r, a, 3) for r in rows):5d} {sum(sc(r, a, k) for r in rows):5d} "
              f"{sum(oracle(r, a, k) for r in rows):9d}")
    for x, y in (('ASQ', 'C'), ('END', 'C'), ('SELF', 'C'), ('ASQM', 'C'),
                 ('ASQ', 'END'), ('ASQ', 'SELF'), ('ASQ', 'ASQM')):
        if x not in shown or y not in shown or not all(has(r, x, k) and has(r, y, k) for r in rows):
            continue
        c = compare(rows, x, y, k)
        print(f"    {x:4s} vs {y:4s}: per-sample {c['pp']:+5.1f} pp "
              f"[template bootstrap {c['ci'][0]:+.1f}, {c['ci'][1]:+.1f}]  perm p={c['p']:.4f}  "
              f"w/o best template {c['loto']:+.1f}\n"
              f"                 rows W={c['w']} L={c['l']} (sign p={c['sign_p']:.3f})  "
              f"templates W={c['tw']} L={c['tl']}  "
              f"SC@{k} W={c['sw']} L={c['sl']} net={c['sw'] - c['sl']:+d}")
    if 'ASQ' in shown and all(has(r, 'ASQ', k) for r in rows):
        fixed = [r for r in rows if acc(r, 'C', k) <= 0.2 + 1e-9 and acc(r, 'ASQ', k) >= 0.6 - 1e-9]
        broken = [r for r in rows if acc(r, 'C', k) >= 0.8 - 1e-9 and acc(r, 'ASQ', k) <= 0.4 + 1e-9]
        print(f"    rows fixed (C <= 1/5 -> ASQ >= 3/5): {len(fixed)}   "
              f"rows broken (C >= 4/5 -> ASQ <= 2/5): {len(broken)}")
        tt = collections.defaultdict(list)
        for r in rows:
            if r['template'] is not None:
                tt[r['template']].append(r)
        worst = sorted(tt.items(), key=lambda kv: sum(acc(r, 'C', k) for r in kv[1]) / len(kv[1]))
        if worst:
            print("    the 8 templates hardest for C (per-sample C -> ASQ):")
            for t, rr in worst[:8]:
                c = 100 * sum(acc(r, 'C', k) for r in rr) / len(rr)
                q = 100 * sum(acc(r, 'ASQ', k) for r in rr) / len(rr)
                print(f"      template {t:2d} ({len(rr)} rows): {c:5.1f}% -> {q:5.1f}%")


def verdicts(main: List[Dict], guard: List[Dict], arms: Sequence[str], k: int) -> List[str]:
    if 'ASQ' not in arms:
        return ['no verdict: the ASQ arm was not requested']
    if not main:
        return ['no verdict: no complete main-set rows']
    out = []
    c = compare(main, 'ASQ', 'C', k)
    facts = (f"{c['pp']:+.1f} pp per sample, permutation p={c['p']:.4f}, "
             f"{c['loto']:+.1f} pp without the most-helped template, "
             f"rows W={c['w']} L={c['l']}, templates W={c['tw']} L={c['tl']}")
    if c['pp'] >= PRIMARY_MIN_PP and c['p'] < PRIMARY_ALPHA and c['loto'] >= PRIMARY_LOTO_PP:
        out.append(f"PRIMARY  SUPPORTED: the Reader improves the Solver's reasoning ({facts})")
    elif c['pp'] <= 0:
        out.append(f"PRIMARY  REFUTED: ASQ <= C ({facts})")
    else:
        out.append(f"PRIMARY  INCONCLUSIVE: {facts}")
    net = c['sw'] - c['sl']
    ok = net >= SYSTEM_MIN_NET and c['sw'] >= 2 * c['sl']
    out.append(f"SYSTEM   SC@{k} ASQ vs C: W={c['sw']} L={c['sl']} net={net:+d} -> "
               + ("the Reader-Solver system beats SC at equal solver samples" if ok
                  else "short of the bar"))
    if 'END' in arms:
        e = compare(main, 'ASQ', 'END', k)['pp']
        ec = compare(main, 'END', 'C', k)['pp']
        if e >= MECH_MIN_PP:
            out.append(f"MECH     the position matters: ASQ - END = {e:+.1f} pp "
                       f"(MathWorld's interleaving effect holds for self-generated notes)")
        elif ec >= MECH_MIN_PP and abs(e) < MECH_MIN_PP:
            out.append(f"MECH     the content matters, not the position: END - C = {ec:+.1f} pp, "
                       f"ASQ - END = {e:+.1f} pp")
        else:
            out.append(f"MECH     undecided: ASQ - END = {e:+.1f} pp, END - C = {ec:+.1f} pp")
    if 'SELF' in arms:
        s = compare(main, 'ASQ', 'SELF', k)['pp']
        out.append(f"MULTI    ASQ - SELF = {s:+.1f} pp -> "
                   + ("a separate Reader beats in-prompt careful reading" if s >= MECH_MIN_PP
                      else "no advantage for a separate Reader shown"))
    if guard:
        g = compare(guard, 'ASQ', 'C', k)
        net = g['sw'] - g['sl']
        out.append(f"GUARD    SC@{k} ASQ vs C net {net:+d} -> "
                   + ("no harm" if net >= GUARD_MIN_NET else "HARM"))
    return out


def complete(rec: Dict, arms: Sequence[str], k: int) -> bool:
    return all(has(rec, a, k) for a in set(arms) | {'C'})


def summarise(recs: List[Dict], arms: Sequence[str], k: int, dev: bool, stub: bool,
              expected: int = 0) -> int:
    rows = [r for r in recs if complete(r, arms, k)]
    if dev:
        for src in sorted({r['source'] for r in rows}):
            for tag in ('main', 'guard'):
                block([r for r in rows if r['source'] == src and r['set'] == tag], arms, k,
                      f"{src.upper()} {tag.upper()}")
    main = [r for r in rows if r['set'] == 'main']
    guard = [r for r in rows if r['set'] == 'guard']
    if not dev or len({r['source'] for r in rows}) > 1:
        block(main, arms, k, 'POOLED MAIN' if dev else 'MAIN')
        block(guard, arms, k, 'POOLED GUARD' if dev else 'GUARD')
    print('\n  pre-registered reading:')
    if expected and len(rows) < expected and not stub:
        print(f"    none: PARTIAL run, {len(rows)}/{expected} rows complete. "
              f"Re-run the identical command to finish.")
        return 0
    if k != 5:
        print(f"    none: k={k} is not the pre-registered k=5 (exploratory run)")
        return 0
    for v in verdicts(main, guard, arms, k):
        print('    ' + v)
    return 0


# ---------------------------------------------------------------------------
# runners
# ---------------------------------------------------------------------------

def _load_prior(path: str, resume: bool, stub: bool) -> Dict[str, Dict]:
    # a stub run never resumes from, or into, the real result files
    if os.path.exists(path) and resume and not (stub and path in (OUT, OUT_DEV)):
        prior = {r['key']: r for r in _read(path).get('rows', [])}
        print(f"resuming: {len(prior)} rows already in {path}")
        return prior
    return {}


def _save(path: str, recs: List[Dict], meta: Dict) -> None:
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(dict(meta, rows=recs), fh, ensure_ascii=False)
    os.replace(tmp, path)


def _line(i: int, n: int, rec: Dict, arms: Sequence[str], k: int, t0: float, ran: int) -> str:
    f = lambda a: f"{a}={sum(V21.correct(x, rec['gold']) for x in answers(rec, a, k))}/{k}"
    shown = ' '.join(f(a) for a in ['C'] + [a for a in arms if a != 'C'] if has(rec, a, k))
    rd = rec.get('reading') or {}
    note = f"notes={rd.get('n_notes', 0)}" + ('' if rd.get('ok', True) else ' READER-FAILED')
    rate = f"{(time.time() - t0) / ran:5.0f}s/row" if ran else ''
    return f"[{i:3d}/{n}] {rec['source']} {rec['set']:5s} {rec['pid'][:24]:24s} {shown}  {note}  {rate}"


def run(args, arms: Sequence[str], dev: bool) -> int:
    _READS.update(n=0, ok=0, shown=False)          # the Reader guard counts per session
    if dev:
        pool = load_pool([s.strip() for s in args.pool.split(',') if s.strip()])
        if args.limit:
            keep = []
            for tag in ('main', 'guard'):
                keep += [r for r in pool if r['set'] == tag][:max(1, args.limit // 2)]
            pool = keep
        if args.k > min(len(r['stored_C']) for r in pool):
            print(f"--k {args.k} exceeds the stored C samples")
            return 2
    else:
        for _, path in MANIFESTS:
            if not os.path.exists(path):
                print(f"missing {path}: run `python pretest_v24.py --build-manifest` first "
                      f"(needs the `datasets` package), then commit the two files.")
                return 2
        pool = load_confirm(args.limit)
    out_path = args.out or (OUT_STUB if args.stub else (OUT_DEV if dev else OUT))
    prior = _load_prior(out_path, args.resume, args.stub)
    recs = []
    for row in pool:
        rec = prior.get(row['key']) or new_record(row)
        rec.setdefault('arms', {})
        if dev:
            rec['arms']['C'] = {'samples': row['stored_C'][:args.k], 'stored': True}
        recs.append((row, rec))
    need = [x for x in recs if not complete(x[1], arms, args.k)]
    print(f"rows: {len(recs)} ({'dev: ' + args.pool if dev else 'confirmation'})  k={args.k}  "
          f"T={args.temperature}  arms={','.join(arms)}  to draw: {len(need)} rows")
    solver = reader = None
    if need:
        solver, reader = build(args, [x[0] for x in need])
    meta = {'mode': 'dev' if dev else 'confirm', 'arms': list(arms), 'k': args.k,
            'temperature': args.temperature, 'max_tokens': args.max_tokens,
            'pool': args.pool if dev else 'v24'}
    t0, ran, stopped = time.time(), 0, False
    deadline = t0 + args.max_hours * 3600 if args.max_hours else 0.0
    touched: List[Dict] = []
    for i, (row, rec) in enumerate(recs, 1):
        if stopped or complete(rec, arms, args.k):
            continue
        ts = time.time()
        try:
            made, stopped = fill(solver, reader, rec, row['text'], arms, args, deadline, dev)
        except RuntimeError:
            # the Reader guard fired: every row drawn this session rests on an
            # unusable reading, i.e. its ASQ/END samples are C samples. Drop
            # them so that a resumed session draws them with a working Reader.
            for rr in touched + [rec]:
                if not rr.get('reading', {}).get('ok', True):
                    rr.pop('reading', None)
                    for a in ('ASQ', 'END'):
                        rr['arms'].pop(a, None)
            _save(out_path, [x[1] for x in recs], meta)
            raise
        touched.append(rec)
        if made:
            ran += 1
            rec['seconds'] = round(rec.get('seconds', 0) + time.time() - ts, 1)
            print(_line(i, len(recs), rec, arms, args.k, t0, ran), flush=True)
            _save(out_path, [x[1] for x in recs], meta)
        if stopped:
            print("\n  --max-hours reached; re-run the identical command to resume.")
    _save(out_path, [x[1] for x in recs], meta)
    print(f"\n  saved: {out_path}")
    return summarise([x[1] for x in recs], arms, args.k, dev=dev, stub=args.stub,
                     expected=len(recs))


# ---------------------------------------------------------------------------
# fresh confirmation rows
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


def build_manifests(seed: int, n_main: int, n_guard: int, force: bool) -> int:
    """100 GSM-Symbolic P2 rows and 20 GSM-Plus distractor rows that no
    earlier version has drawn. Written once; the run only reads them."""
    for _, path in MANIFESTS:
        if os.path.exists(path) and not force:
            print(f"{path} exists; refusing to overwrite pre-registered rows (--force to redo)")
            return 1
    from datasets import load_dataset
    used = set()
    for p in EARLIER_MANIFESTS:
        if os.path.exists(p):
            used |= {r['problem_id'] for r in _read(p)['rows']}
    print(f"excluding {len(used)} rows drawn by earlier versions")
    ds = load_dataset('apple/GSM-Symbolic', name='p2', split='test')
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    main = []
    for i in idx:
        pid = f'gsm-symbolic_p2_{i}'
        g, text = _gold(ds[i].get('answer')), (ds[i].get('question') or '').strip()
        if pid in used or g is None or len(text) < 20:
            continue
        main.append({'problem_id': pid, 'text': text, 'gold': g, 'dataset': 'gsm-symbolic:p2'})
        if len(main) >= n_main:
            break
    gs = load_dataset('qintongli/GSM-Plus', split='test')
    gidx = [i for i in range(len(gs)) if gs[i]['perturbation_type'] == 'distraction insertion']
    random.Random(seed).shuffle(gidx)
    guard = []
    for i in gidx:
        pid = f'gsm-plus_{i}'
        try:
            g = float(str(gs[i]['answer']).replace(',', '').strip())
        except ValueError:
            continue
        if pid in used:
            continue
        guard.append({'problem_id': pid, 'text': gs[i]['question'].strip(), 'gold': g,
                      'dataset': 'gsm-plus:distraction insertion'})
        if len(guard) >= n_guard:
            break
    for (tag, path), rows, name in ((MANIFESTS[0], main, 'gsm-symbolic:p2'),
                                    (MANIFESTS[1], guard, 'gsm-plus:distraction insertion')):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump({'dataset': name, 'seed': seed, 'n': len(rows),
                       'note': f'v24 confirmation {tag} rows: fresh seed, disjoint from every '
                               f'row of v20-v23',
                       'rows': rows}, fh, ensure_ascii=False, indent=1)
        print(f"wrote {len(rows)} {tag} rows -> {path}")
    return 0 if len(main) == n_main and len(guard) == n_guard else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dev', action='store_true',
                    help='stored rows with their C samples (no C is drawn)')
    ap.add_argument('--pool', default='v22', help='dev pool: v22, or v21,v22')
    ap.add_argument('--arms', default='', help=f'comma list from {",".join(ARMS)}')
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--out', default='')
    ap.add_argument('--preset', default='qwen_math7b_mixed')
    ap.add_argument('--temperature', type=float, default=0.8)
    ap.add_argument('--max-tokens', type=int, default=1024)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--stub', action='store_true')
    ap.add_argument('--no-resume', dest='resume', action='store_false')
    ap.add_argument('--max-hours', type=float, default=0.0)
    ap.add_argument('--summary-only', action='store_true')
    ap.add_argument('--build-manifest', action='store_true')
    ap.add_argument('--seed', type=int, default=CONFIRM_SEED)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    if args.build_manifest:
        return build_manifests(args.seed, 100, 20, args.force)
    spec = args.arms or (DEFAULT_DEV_ARMS if args.dev else DEFAULT_CONFIRM_ARMS)
    arms = [a.strip().upper() for a in spec.split(',') if a.strip()]
    bad = [a for a in arms if a not in ARMS]
    if bad or not arms:
        print(f"unknown arm(s) {bad}; choose from {ARMS}")
        return 2
    if args.summary_only:
        data = _read(args.out or (OUT_DEV if args.dev else OUT))
        # an explicit --arms reads just those arms, so a PRIMARY that is complete
        # can be read while a later control arm is still partial
        if not args.arms:
            arms = [a for a in ARMS if a in data.get('arms', arms)]
        return summarise(data['rows'], arms, data.get('k', args.k), dev=args.dev, stub=args.stub,
                         expected=len(data['rows']))
    return run(args, arms, dev=args.dev)


if __name__ == '__main__':
    sys.exit(main())
