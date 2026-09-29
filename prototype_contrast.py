"""
[v26.0] Prototype-Contrastive Decoding (PCD): make the Solver reason about
the problem that is written, not the familiar one it recites.

THE PROBLEM IT IS BUILT FOR (hand taxonomy, 2026-09-29)
-------------------------------------------------------
On the 18 P2 rows where none of 5 plain samples is right (v22 seed 45), 12
are an explicit ADDED or MODIFIED condition that the Solver overrides with
the familiar version of the problem. GSM-Symbolic P2 is built by adding
clauses to GSM8K problems, and the errors sit exactly on those clauses:
  - "(including the plants width)": width + gap is used anyway (3 rows);
  - "these 3 groups ... 3 cheerleaders accompany each": 2 groups counted (4);
  - "16 white animals, half of which were rabbits": the other 8 dropped (2);
  - post-injury "60 minutes on the beach" applied to the pre-injury speed;
  - an added return trip priced at one leg instead of the whole path.
The other 6 are gold/ambiguity (4) and arithmetic slips (2).
Yan et al. (RoR-Bench, arXiv 2504.00509) call this "recitation over
reasoning": top models lose ~60% when one phrase changes. Their inference-
time fixes (notice prompts, modified few-shots) "can mitigate the performance
drop slightly, they are far from satisfactory and a more complete solution
is still yet to be proposed"; showing the original problem makes it worse.

THE METHOD
----------
1. A Prototype agent (the Reader model, Qwen2.5-7B-Instruct, greedy) rewrites
   the problem as its most familiar version: same sentences, names, numbers
   and question, with the unusual twist removed. That is the problem the
   Solver is likely to recite: an explicit "null hypothesis".
2. The Solver (Qwen2.5-Math-7B-Instruct) samples its CoT for the REAL problem,
   but every token is drawn from
        (1 + a) * log p(y | real, y<t)  -  a * log p(y | prototype, y<t)
   restricted to tokens with p(y | real) >= b * max p(y | real)
   (the adaptive plausibility constraint of Contrastive Decoding). Both
   contexts share the generated prefix. Where the two texts agree the two
   distributions agree and nothing changes; where the real text changes the
   situation, its effect is amplified and the recited template is damped.
   The constraint means PCD only re-ranks tokens the Solver already finds
   plausible for the real problem.
a = 1.0 and b = 0.1 are fixed in advance: CAD's knowledge-conflict setting
(Shi et al., NAACL 2024) and Contrastive Decoding's plausibility threshold
(Li et al., ACL 2023). Nothing is tuned on P2.

WHAT IS AND IS NOT NEW (checked 2026-09-29)
-------------------------------------------
Not new: contrasting two conditional distributions. CAD contrasts with vs
without the context (summarization, knowledge-conflict QA: single-step
outputs); Contrastive Decoding contrasts an expert with an amateur MODEL;
instructive decoding contrasts with noisy instructions; source-contrastive
MT contrasts with a random other source.
New here, as far as the searches found: the contrast input is a
counterfactual CANONICAL version of the same problem, written by a second
agent, used to suppress recitation in multi-step reasoning. RoR-Bench names
exactly this gap as open.

Everything below except `sample_streams` is model-free. `sample_streams` needs
torch and a HF causal LM; test_v26.py checks it against `generate` on a tiny
random Qwen2 model.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import situation_reader as SR

ALPHA = 1.0          # CAD's knowledge-conflict setting
ALPHA_HALF = 0.5     # the dose-response arm
BETA = 0.1           # Contrastive Decoding's plausibility threshold
PROTO_MAX_TOKENS = 600

# ---------------------------------------------------------------------------
# the Prototype agent
# ---------------------------------------------------------------------------

PROTO_SYSTEM = (
    "You rewrite math word problems into their most ordinary, familiar version. "
    "Copy the problem sentence by sentence, keeping every name, number and the final "
    "question exactly as written, but remove anything that makes it unusual: an exception "
    "or special condition, a clarification in parentheses, a change that happens partway "
    "through the story, an extra person, group or step added to the usual story. "
    "Never solve the problem and never add anything new. If nothing is unusual, copy the "
    "problem unchanged. Output only the rewritten problem.")

# Written for this purpose; none is a GSM8K or GSM-Symbolic problem.
PROTO_DEMOS = [
    ("Mia buys 4 packs of pens with 6 pens in each pack. She gives 5 pens to her brother, "
     "but her brother gives 2 of them back the next day. How many pens does Mia have now?",
     "Mia buys 4 packs of pens with 6 pens in each pack. She gives 5 pens to her brother. "
     "How many pens does Mia have now?"),
    ("A tank holds 120 liters. A pipe fills it at 8 liters per minute. After the first 5 "
     "minutes, a leak starts that drains 2 liters per minute. How many minutes does it take "
     "to fill the empty tank?",
     "A tank holds 120 liters. A pipe fills it at 8 liters per minute. How many minutes does "
     "it take to fill the empty tank?"),
    ("A fence is 60 meters long. A post is placed every 5 meters (the 5 meters include the "
     "post itself). How many posts are needed?",
     "A fence is 60 meters long. A post is placed every 5 meters. How many posts are needed?"),
    ("Leo reads 12 pages every day. How many pages does he read in 3 weeks?",
     "Leo reads 12 pages every day. How many pages does he read in 3 weeks?"),
]


def proto_messages(text: str) -> List[Dict[str, str]]:
    msgs = [{'role': 'system', 'content': PROTO_SYSTEM}]
    for problem, proto in PROTO_DEMOS:
        msgs.append({'role': 'user', 'content': f"Problem: {problem}"})
        msgs.append({'role': 'assistant', 'content': proto})
    msgs.append({'role': 'user', 'content': f"Problem: {text}"})
    return msgs


_PREFIX = re.compile(r'^\s*(?:\*\*)?\s*(?:rewritten|familiar|ordinary|standard)?\s*'
                     r'(?:problem|version)\s*(?:\*\*)?\s*:\s*', re.I)
_SOLVING = re.compile(r'\b(answer\s*[:=]|let\'s solve|step 1|\\boxed|therefore,? the)\b', re.I)


def norm(s: str) -> str:
    return re.sub(r'\s+', ' ', str(s or '')).strip()


def question_of(text: str) -> str:
    """The question: the last sentence with a '?', else the last sentence."""
    sents = SR.sentences(text)
    qs = [s for s in sents if '?' in s]
    return (qs[-1] if qs else (sents[-1] if sents else str(text))).strip()


def parse_prototype(raw: str, text: str) -> Dict:
    """Sanitise the Prototype agent's output. A failed prototype becomes the
    real text itself, which makes PCD exactly the plausibility-constrained
    sampler on that row (and the summary counts it)."""
    out = {'raw': str(raw or '')[:3000]}
    s = str(raw or '').strip().strip('`').strip()
    s = _PREFIX.sub('', s).strip().strip('"').strip()
    # the rewritten problem is one paragraph; anything after a blank line is
    # the model explaining what it removed, which must not enter the contrast
    s = s.split('\n\n')[0].strip()
    reason = ''
    if not s:
        reason = 'empty'
    elif _SOLVING.search(s):
        reason = 'solved instead of rewriting'
    elif len(s) > 1.3 * len(text) + 40:
        reason = 'longer than the problem'
    elif len(s) < 0.25 * len(text):
        reason = 'too short'
    if reason:
        out.update(text=text, ok=False, reason=reason, identical=True)
    else:
        out.update(text=s, ok=True, reason='', identical=norm(s) == norm(text))
    out['kept_question'] = question_of(text) in out['text']
    out['ratio'] = round(len(out['text']) / max(1, len(text)), 3)
    return out


# ---------------------------------------------------------------------------
# the scorer: pure functions over logits (torch tensors)
# ---------------------------------------------------------------------------

@dataclass
class Stream:
    """One sample being generated. `main` and `contrast` index the prompt
    rows; both rows receive every token the stream samples."""
    arm: str
    main: int
    contrast: Optional[int] = None
    alpha: float = 0.0
    beta: float = 0.0


def combine(main_logits, contrast_logits, alpha, beta):
    """[S,V] logits -> [S,V] scores in log space.
    alpha/beta are [S] tensors. alpha=0 and beta=0 give log_softmax(main),
    i.e. plain sampling, exactly."""
    import torch
    lp = torch.log_softmax(main_logits.float(), dim=-1)
    lc = torch.log_softmax(contrast_logits.float(), dim=-1)
    a = alpha.to(lp.dtype).unsqueeze(-1)
    s = (1.0 + a) * lp - a * lc
    b = beta.to(lp.dtype).unsqueeze(-1)
    cut = lp.max(dim=-1, keepdim=True).values + torch.log(b.clamp(min=1e-30))
    implausible = (b > 0) & (lp < cut)
    return s.masked_fill(implausible, float('-inf'))


def warp(scores, temperature: float, top_k: int, top_p: float):
    """Temperature -> top-k -> top-p, as HF's warpers, then renormalise."""
    import torch
    s = scores / temperature
    if top_k and top_k > 0:
        k = min(int(top_k), s.size(-1))
        kth = torch.topk(s, k, dim=-1).values[..., -1:]
        s = s.masked_fill(s < kth, float('-inf'))
    if top_p is not None and top_p < 1.0:
        srt, idx = torch.sort(s, descending=False, dim=-1)
        cum = srt.softmax(dim=-1).cumsum(dim=-1)
        remove = cum <= (1 - top_p)
        remove[..., -1:] = False
        s = s.masked_fill(remove.scatter(-1, idx, remove), float('-inf'))
    return torch.log_softmax(s, dim=-1)


# ---------------------------------------------------------------------------
# the sampler
# ---------------------------------------------------------------------------

def _logits_to_keep_kw(model) -> Dict[str, int]:
    """transformers 4.50 calls it logits_to_keep, 4.45-4.49 num_logits_to_keep
    in some classes; pass whichever this model's forward accepts."""
    import inspect
    try:
        params = inspect.signature(type(model).forward).parameters
    except (TypeError, ValueError):
        return {}
    for name in ('logits_to_keep', 'num_logits_to_keep'):
        if name in params:
            return {name: 1}
    return {}


def sample_streams(model, prompts: Sequence[Sequence[int]], streams: Sequence[Stream], *,
                   max_new_tokens: int, temperature: float, top_k: int, top_p: float,
                   eos_ids: Sequence[int], pad_id: int, seed: int = 0) -> List[List[int]]:
    """Decode every stream in one left-padded batch with a shared KV cache.
    Each prompt row belongs to exactly one stream. A finished stream's rows
    are dropped from the batch and the cache. temperature <= 0 is greedy.
    Returns the generated token ids per stream (EOS included if reached)."""
    import torch
    rows_of = [[s.main] + ([s.contrast] if s.contrast is not None else []) for s in streams]
    owner = {}
    for j, rr in enumerate(rows_of):
        for r in rr:
            if r in owner:
                raise ValueError(f"prompt row {r} is used by two streams")
            owner[r] = j
    if sorted(owner) != list(range(len(prompts))):
        raise ValueError("every prompt row must belong to exactly one stream")
    dev = next(model.parameters()).device
    B, L = len(prompts), max(len(p) for p in prompts)
    ids = torch.full((B, L), int(pad_id), dtype=torch.long)
    attn = torch.zeros((B, L), dtype=torch.long)
    for i, p in enumerate(prompts):
        ids[i, L - len(p):] = torch.tensor(list(p), dtype=torch.long)
        attn[i, L - len(p):] = 1
    pos = (attn.cumsum(-1) - 1).masked_fill(attn == 0, 1)
    ids, attn, pos = ids.to(dev), attn.to(dev), pos.to(dev)
    eos = set(int(e) for e in eos_ids)

    # only the last position's logits are needed; the full prefill logits of
    # ~24 rows x ~300 tokens x a 152k vocabulary would not fit on a T4
    keep_kw = _logits_to_keep_kw(model)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=attn, position_ids=pos, use_cache=True, **keep_kw)
    cache, logits = out.past_key_values, out.logits[:, -1, :]
    # A model split over two GPUs keeps its cache layers on both and may return
    # logits on either: sampling lives on the logits' device, and batch
    # pruning uses CPU indices, which PyTorch accepts for tensors on any device.
    ldev = logits.device
    gen = torch.Generator(device=ldev)
    gen.manual_seed(int(seed))
    alpha = torch.tensor([float(s.alpha) for s in streams], device=ldev)
    beta = torch.tensor([float(s.beta) for s in streams], device=ldev)
    alive = list(range(B))                      # batch position -> prompt row
    last_pos = {r: int(pos[r, -1]) for r in range(B)}
    tokens: List[List[int]] = [[] for _ in streams]
    done = [False] * len(streams)
    for _ in range(max_new_tokens):
        at = {r: i for i, r in enumerate(alive)}
        act = [j for j in range(len(streams)) if not done[j]]
        mi = torch.tensor([at[streams[j].main] for j in act], device=ldev)
        ci = torch.tensor([at[streams[j].contrast if streams[j].contrast is not None
                              else streams[j].main] for j in act], device=ldev)
        sc = combine(logits[mi], logits[ci], alpha[act], beta[act])
        if temperature is None or temperature <= 0:
            nxt = sc.argmax(dim=-1)
        else:
            probs = warp(sc, temperature, top_k, top_p).exp()
            nxt = torch.multinomial(probs, 1, generator=gen).squeeze(-1)
        for j, t in zip(act, nxt.tolist()):
            tokens[j].append(int(t))
            if int(t) in eos:
                done[j] = True
        if all(done):
            break
        need = sorted(r for j in range(len(streams)) if not done[j] for r in rows_of[j])
        if len(need) < len(alive):
            keep = torch.tensor([at[r] for r in need])          # CPU on purpose
            cache.batch_select_indices(keep)
            attn = attn[keep.to(attn.device)]
            alive = need
        step_ids = torch.tensor([[tokens[owner[r]][-1]] for r in alive], device=dev)
        for r in alive:
            last_pos[r] += 1
        step_pos = torch.tensor([[last_pos[r]] for r in alive], device=dev)
        attn = torch.cat([attn, torch.ones((len(alive), 1), dtype=attn.dtype, device=dev)], dim=-1)
        with torch.no_grad():
            out = model(input_ids=step_ids, attention_mask=attn, position_ids=step_pos,
                        past_key_values=cache, use_cache=True)
        cache, logits = out.past_key_values, out.logits[:, -1, :].to(ldev)
    return tokens


# ---------------------------------------------------------------------------
# one row: the arms
# ---------------------------------------------------------------------------

# arm -> (contrast context, alpha, beta). L0 is exactly the plain sampler of
# every earlier version; MINP isolates the plausibility constraint; CADQ is a
# generic contrast (question only) to show the PROTOTYPE is what matters.
ARMS = {
    'L0': (None, 0.0, 0.0),
    'MINP': (None, 0.0, BETA),
    'PCD': ('prototype', ALPHA, BETA),
    'PCD5': ('prototype', ALPHA_HALF, BETA),
    'CADQ': ('question', ALPHA, BETA),
}
DEFAULT_ARMS = ('L0', 'MINP', 'PCD', 'PCD5', 'CADQ')


def layout(arms: Sequence[str], k: int, contexts: Dict[str, str]):
    """Prompt contents and streams for one row: k samples per arm. `contexts`
    maps 'real' / 'prototype' / 'question' to problem texts."""
    prompts: List[str] = []
    streams: List[Stream] = []
    for arm in arms:
        con, a, b = ARMS[arm]
        for _ in range(k):
            m = len(prompts)
            prompts.append(SR.cot_prompt(contexts['real']))
            c = None
            if con is not None:
                c = len(prompts)
                prompts.append(SR.cot_prompt(contexts[con]))
            streams.append(Stream(arm=arm, main=m, contrast=c, alpha=a, beta=b))
    return prompts, streams
