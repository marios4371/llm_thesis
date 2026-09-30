"""
[v28.0] Context-DPO against recitation: teach the Solver to prefer a solution
of THIS problem over the solution of its familiar version.

Why (V28_PLAN.md has the numbers)
---------------------------------
On GSM-Symbolic P2 the right answer is already in the Solver's samples on
91-95 of 100 rows, but selection tops out at 81. The verifier-level ties are
decided by a rule that is already right on 84% of the disputes, and the
errors it loses are explicit misreadings the PRM scores 1.0 ("2 groups" when
the text says "These 3 groups"). Every selection idea needs a judge that
beats 84% on those disputes, and every same-family judge recites. So v28
changes the Solver instead of choosing among its outputs: the thesis's first
weight update.

The published idea and its open window
--------------------------------------
Context-DPO (Bi et al., ACL Findings 2025) aligns a model to its context when
the context conflicts with what it memorised. The preference pair is a
"faithful" answer (follows the counterfactual context) against a "stubborn"
one (the parametric answer). Its Limitations restrict it to factual
knowledge-conflict QA. Recitation over reasoning (RoR-Bench, arXiv
2504.00509) is the procedural form of the same conflict: a memorised solution
schema overrides a modified condition. RoR-Bench tried only prompts (+3 to
+12 points, "far from satisfactory"); MATH-Perturb proposes no fix. v26
already tried the DECODING-time answer from the knowledge-conflict literature
(CAD). It failed because its truncation removed the low-probability branches
where the right solutions live. DPO is the TRAINING-time counterpart: it
moves probability mass and truncates nothing at inference.

The pair (built by the multi-agent pipeline, no human in the loop)
------------------------------------------------------------------
  prompt    the problem x, in the CoT prompt of every earlier version
  chosen    a right solution of x: a stored plain sample graded against gold
            (training folds only)
  rejected  arm FV: the Solver's own solution of the familiar version x0
            (the Prototype agent's rewrite of x, twist removed), whose answer
            is wrong for x. This is the stubborn, recited response.
            arm RW (control): a stored wrong sample of x, matched row by row
            and in number. It isolates whether the familiar-version
            construction matters, or any wrong sample would do.
The loss is DPO plus an NLL term on the chosen response, as in Iterative RPO
(Pang et al. 2024); without it, DPO on long reasoning tends to lower the
chosen likelihood too.

This module is pure: no torch at import. Tensors are handled by duck typing
in dpo_terms, so the same code is checked offline with floats.
"""
from __future__ import annotations

import collections
import math
import random
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pretest_v21 as V21

# ---------------------------------------------------------------------------
# pre-registered constants (frozen 2026-09-30, before any adapter exists)
# ---------------------------------------------------------------------------

FOLD_SEED = 28
MAX_CHOSEN = 2              # right solutions per row
MAX_NEG = 2                 # rejected solutions per row (same count in FV and RW)
K_FAMILIAR = 2              # Solver samples on the prototype x0
RAW_CAP = 3990              # the stored raw texts were cut at 4000 characters
MAX_LEN_TOKENS = 1536       # prompt + response; longer pairs are dropped, not cut

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')
BETA = 0.1                  # DPO temperature (the DPO paper's default)
NLL_WEIGHT = 1.0            # Iterative RPO's weight on the chosen NLL
LR = 1e-4                   # LoRA, few optimizer steps (about 20-40 per worker)
EPOCHS = 2
ACCUM = 4                   # pairs per optimizer step
WARMUP_FRAC = 0.1
MAX_GRAD_NORM = 1.0
TRAIN_SEED = 28

ARMS = ('FV', 'RW')         # FV = familiar-version negatives (the method); RW = control


# ---------------------------------------------------------------------------
# samples
# ---------------------------------------------------------------------------

def as_float(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def usable(sample: Dict) -> bool:
    """A complete solution with a parsed answer. Texts at the 4000-character
    storage cap were cut mid-solution and must not be taught as a response."""
    raw = str(sample.get('raw') or '')
    return (as_float(sample.get('answer')) is not None and raw.strip() != ''
            and len(raw) < RAW_CAP)


def right(sample: Dict, gold) -> bool:
    return V21.correct(as_float(sample.get('answer')), as_float(gold))


def _norm(s: str) -> str:
    return re.sub(r'\s+', ' ', str(s or '')).strip()


def dedup(samples: Iterable[Dict]) -> List[Dict]:
    seen, out = set(), []
    for s in samples:
        k = _norm(s.get('raw', ''))
        if k and k not in seen:
            seen.add(k)
            out.append(s)
    return out


def edited(prototype: Optional[Dict]) -> bool:
    """A usable prototype that actually removed something."""
    return bool(prototype) and bool(prototype.get('ok')) and not prototype.get('identical', True)


# ---------------------------------------------------------------------------
# folds: templates, never rows
# ---------------------------------------------------------------------------

def split_folds(rows: Sequence[Dict], seed: int = FOLD_SEED) -> Dict[int, int]:
    """template -> fold (0/1). Templates are shuffled with a fixed seed and
    each goes to the fold with fewer rows so far (ties to fold 0). Instances
    of one template never sit on both sides: the test is transfer to twists
    the model was not trained on."""
    count = collections.Counter(r['template'] for r in rows if r.get('template') is not None)
    temps = sorted(count)
    random.Random(seed).shuffle(temps)
    load = [0, 0]
    out: Dict[int, int] = {}
    for t in temps:
        f = 0 if load[0] <= load[1] else 1
        out[t] = f
        load[f] += count[t]
    return out


def split_list(keys: Sequence[str], seed: int) -> Dict[str, int]:
    """key -> half (0/1), alternating over a seeded shuffle (the guard rows)."""
    ks = sorted(keys)
    random.Random(seed).shuffle(ks)
    return {k: i % 2 for i, k in enumerate(ks)}


# ---------------------------------------------------------------------------
# pairs
# ---------------------------------------------------------------------------

def familiar_answer(familiar: Sequence[Dict], gold) -> Optional[float]:
    """The recited answer: the most frequent answer among the usable samples
    on the prototype that is WRONG for the real problem (ties: first seen).
    None when the familiar version gives the real answer, or nothing usable."""
    wrong = [as_float(s['answer']) for s in familiar if usable(s) and not right(s, gold)]
    if not wrong:
        return None
    c = collections.Counter(wrong)
    top = max(c.values())
    return next(a for a in wrong if c[a] == top)


def candidates(row: Dict) -> Dict[str, List[Dict]]:
    """The three pools of one row, each in a fixed order."""
    gold = row['gold']
    chosen = dedup([s for s in list(row.get('base', [])) + list(row.get('extra', []))
                    if usable(s) and right(s, gold)])
    fv = dedup([s for s in row.get('familiar', []) if usable(s) and not right(s, gold)])
    rw = dedup([s for s in row.get('base', []) if usable(s) and not right(s, gold)])
    return {'chosen': chosen, 'FV': fv, 'RW': rw}


def eligible(row: Dict) -> bool:
    c = candidates(row)
    return edited(row.get('prototype')) and bool(c['chosen']) and bool(c['FV']) and bool(c['RW'])


def build_pairs(row: Dict, arm: str) -> List[Dict]:
    """Matched pairs: FV and RW get the same rows, the same chosen responses
    and the same number of rejected ones; only the rejected text differs."""
    if arm not in ARMS:
        raise ValueError(f'unknown arm {arm}')
    if not eligible(row):
        return []
    c = candidates(row)
    n_c = min(len(c['chosen']), MAX_CHOSEN)
    n_n = min(len(c['FV']), len(c['RW']), MAX_NEG)
    out = []
    for i in range(n_c):
        for j in range(n_n):
            out.append({'key': row['key'], 'pid': row['pid'], 'template': row.get('template'),
                        'arm': arm, 'problem': row['text'],
                        'chosen': c['chosen'][i]['raw'], 'rejected': c[arm][j]['raw'],
                        'chosen_answer': as_float(c['chosen'][i]['answer']),
                        'rejected_answer': as_float(c[arm][j]['answer'])})
    return out


def prompt_content(problem: str) -> str:
    """The user turn of every earlier version (pretest_v21.COT_PROMPT)."""
    return V21.COT_PROMPT.format(problem=problem)


# ---------------------------------------------------------------------------
# the loss (floats or torch tensors)
# ---------------------------------------------------------------------------

def _softplus(x):
    if hasattr(x, 'detach'):
        import torch.nn.functional as F
        return F.softplus(x)
    return max(x, 0.0) + math.log1p(math.exp(-abs(x)))


def dpo_terms(pol_c, pol_r, ref_c, ref_r, n_tok_c, beta: float = BETA,
              nll_weight: float = NLL_WEIGHT) -> Dict:
    """Sequence log-probs in, loss out.
      margin = beta * ((pol_c - ref_c) - (pol_r - ref_r))
      dpo    = -log sigmoid(margin) = softplus(-margin)
      nll    = -pol_c / n_tok_c                  (mean token NLL of chosen)
      loss   = dpo + nll_weight * nll"""
    margin = beta * ((pol_c - ref_c) - (pol_r - ref_r))
    dpo = _softplus(-margin)
    nll = -pol_c / max(1, int(n_tok_c))
    return {'loss': dpo + nll_weight * nll, 'dpo': dpo, 'nll': nll, 'margin': margin}


def lr_at(step: int, total: int, base: Optional[float] = None,
          warmup_frac: float = WARMUP_FRAC) -> float:
    """Linear warm-up over the first warmup_frac of optimizer steps, then
    linear decay to zero at the last step. `base` defaults to LR, read at
    call time."""
    base = LR if base is None else base
    total = max(1, int(total))
    warm = max(1, int(round(warmup_frac * total)))
    if step < warm:
        return base * (step + 1) / warm
    return base * max(0.0, (total - step) / max(1, total - warm))


def epoch_order(n: int, epoch: int, seed: int = TRAIN_SEED) -> List[int]:
    idx = list(range(n))
    random.Random(seed * 1000 + epoch).shuffle(idx)
    return idx


# ---------------------------------------------------------------------------
# encoding (any tokenizer with apply_chat_template / __call__)
# ---------------------------------------------------------------------------

def encode(tok, problem: str, response: str, end_id: Optional[int]) -> Tuple[List[int], int]:
    """(ids, n_prompt). The prompt goes through the chat template exactly as
    the samples were drawn; the response is tokenised on its own and closed
    with the end-of-turn token, so the model is also taught to stop."""
    prompt = tok.apply_chat_template([{'role': 'user', 'content': prompt_content(problem)}],
                                     tokenize=False, add_generation_prompt=True)
    p_ids = list(tok(prompt, add_special_tokens=False)['input_ids'])
    r_ids = list(tok(response, add_special_tokens=False)['input_ids'])
    if end_id is not None:
        r_ids.append(int(end_id))
    return p_ids + r_ids, len(p_ids)


def encode_pair(tok, pair: Dict, end_id: Optional[int], max_len: int = MAX_LEN_TOKENS) -> Optional[Dict]:
    c_ids, n_p = encode(tok, pair['problem'], pair['chosen'], end_id)
    r_ids, n_p2 = encode(tok, pair['problem'], pair['rejected'], end_id)
    if n_p != n_p2 or c_ids[:n_p] != r_ids[:n_p]:
        raise RuntimeError('chosen and rejected got different prompts')
    if len(c_ids) > max_len or len(r_ids) > max_len:
        return None
    return {'c_ids': c_ids, 'r_ids': r_ids, 'n_prompt': n_p,
            'n_tok_c': len(c_ids) - n_p, 'n_tok_r': len(r_ids) - n_p}


# ---------------------------------------------------------------------------
# reading the results
# ---------------------------------------------------------------------------

def per_sample(samples: Sequence[Dict], gold) -> Optional[float]:
    if not samples:
        return None
    return sum(right(s, gold) for s in samples) / len(samples)


def vote(samples: Sequence[Dict]) -> Optional[float]:
    """Plurality over parsed answers (ties: first seen), the SC rule."""
    ans = [as_float(s.get('answer')) for s in samples]
    ans = [a for a in ans if a is not None]
    if not ans:
        return None
    groups: List[List[float]] = []
    for a in ans:
        for g in groups:
            if V21.correct(a, g[0]):
                g.append(a)
                break
        else:
            groups.append([a])
    top = max(len(g) for g in groups)
    return next(g[0] for g in groups if len(g) == top)


def recites(samples: Sequence[Dict], familiar: Optional[float]) -> Optional[float]:
    """Share of samples whose answer is the familiar version's answer."""
    if familiar is None or not samples:
        return None
    return sum(V21.correct(as_float(s.get('answer')), familiar) for s in samples) / len(samples)


def perm_p(diffs: Sequence[float], b: int = 10000, seed: int = 0) -> float:
    """Two-sided sign-flip permutation test over rows."""
    d = [float(x) for x in diffs]
    if not d or all(abs(x) < 1e-12 for x in d):
        return 1.0
    obs = abs(sum(d))
    rng = random.Random(seed)
    hit = 0
    for _ in range(b):
        s = sum(x if rng.random() < 0.5 else -x for x in d)
        hit += abs(s) >= obs - 1e-12
    return (hit + 1) / (b + 1)


def template_ci(diffs: Sequence[float], templates: Sequence, b: int = 2000,
                seed: int = 0) -> Tuple[float, float]:
    """95% bootstrap interval (pp) resampling whole templates."""
    by: Dict = collections.defaultdict(list)
    for v, t in zip(diffs, templates):
        by[t].append(v)
    keys = list(by)
    if not keys:
        return (0.0, 0.0)
    rng = random.Random(seed)
    boots = []
    for _ in range(b):
        vals = [v for kk in (rng.choice(keys) for _ in keys) for v in by[kk]]
        boots.append(100 * sum(vals) / len(vals))
    boots.sort()
    return boots[int(0.025 * b)], boots[int(0.975 * b) - 1]
