"""
[v20.0] Meaning-preserving RE-RENDERINGS of a word problem.

WHY THIS MODULE EXISTS
----------------------
Every multi-agent system in the literature draws its agent diversity from the
OUTPUT space: sampling temperature, a second base model, or a role prompt. In
this project's actual deployment regime -- one local 7B with
`do_sample=False` -- that space is empty, and that is a mechanical explanation
for why every selection layer here measured null (see the v15.8 / v16 notes in
Mas_solver.py) and for why published multi-agent debate does not beat plain
chain-of-thought. The named, unsolved limitation in that literature is agent
homogeneity: agents on one model make correlated errors.

This module supplies the other axis. The robustness literature has measured,
repeatedly, that an LLM's answer to a math problem moves when the problem is
restated without changing its meaning -- and has only ever reported that as a
defect. FormInv states outright that it proposes no inference-time method and
does not aggregate across paraphrases. The two closest method papers each name
multi-step reasoning as future work: Raiyan et al. (ParaMAWPS) vote over
paraphrases but with DeBERTa on single-equation problems, and PCS scores
metamorphic consistency at inference but is classification-only. So the
variance is documented, the mechanism is proven elsewhere, and nobody has used
it here.

THE CONSTRUCTION IS THE RISK, NOT THE IDEA
------------------------------------------
v16.2 and v19 both died on construction, not mechanism: a rescaled twin that
sent a quantity through zero, or turned whole riders into 1.75 people, is not
the same problem, and no downstream cleverness recovers that. The lesson is
encoded here as a hard rule: **a rendering that is not provably faithful is
refused, not repaired.** `is_faithful` requires the multiset of numeric
literals to be preserved exactly, and every transform either returns a
rendering that passes it or returns `ok=False` with a reason. A refused
rendering falls back to the identity, which is what a deployed system would do.

The structural transforms are pure re-layout: they move whole sentences and
add headings, and they never touch a character inside a sentence. That makes
their faithfulness a property of the code rather than a hope about a model,
and it is why the shipped set is made of them. `paraphrase` is the one
transform that goes through the LLM; it is validated hardest, and the
2026-09-22 smoke test refused it on 2 of 2 rows, which is why it is no longer
in DEFAULT_SET. See the note there.

NOT IN THE TRANSFORM SET, DELIBERATELY
--------------------------------------
Numeral re-formatting -- comma grouping, "3.47 million", spelled place value.
The published result that commas fix LLM arithmetic is about tokenizers that
chunk digit runs left-to-right in groups of up to three (cl100k and friends).
Qwen2.5-Math's pre-tokenizer alternates on bare `\\p{N}`: **one digit per
token**, already perfectly place-value aligned. That failure mode does not
exist for this model, so the transform can only add tokens and cannot buy
accuracy. Checked against the published tokenizer.json, 2026-09-22.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# A numeric literal: optional currency/sign, digits with optional thousands
# separators, optional decimal tail, optional trailing percent.
NUMBER_RE = re.compile(r'\d[\d,]*\.?\d*')

# Periods that are NOT sentence ends. Protected before splitting, restored
# after, so that "Mr. Smith paid $3.50" stays one sentence.
_ABBREV = re.compile(
    r'\b(Mr|Mrs|Ms|Dr|Prof|St|Jr|Sr|vs|etc|approx|Inc|Ltd|No|Fig|Ave|Rd)\.',
    re.I)
_DECIMAL = re.compile(r'(\d)\.(\d)')
_DOT = '\x00DOT\x00'


def numbers(text: str) -> Dict[float, int]:
    """The multiset of numeric literals in `text`, comma-normalised.

    This is the invariant every rendering must preserve. It is deliberately
    strict: a paraphrase that writes "two" where the original wrote "2" is
    refused rather than accepted, because the cheap way to be wrong here is to
    accept a rendering that quietly dropped a quantity.
    """
    out: Dict[float, int] = {}
    for tok in NUMBER_RE.findall(text):
        tok = tok.replace(',', '').rstrip('.')
        if not tok:
            continue
        try:
            v = float(tok)
        except ValueError:
            continue
        out[v] = out.get(v, 0) + 1
    return out


def split_sentences(text: str) -> List[str]:
    """Sentence split that survives decimals, currency and abbreviations."""
    protected = _DECIMAL.sub(lambda m: m.group(1) + _DOT + m.group(2), text)
    protected = _ABBREV.sub(lambda m: m.group(1) + _DOT, protected)
    parts = re.split(r'(?<=[.!?])\s+', protected)
    return [p.replace(_DOT, '.').strip() for p in parts if p.strip()]


def _question_index(sents: List[str]) -> int:
    """Index of the interrogative sentence, or the last one if none asks."""
    for i in range(len(sents) - 1, -1, -1):
        if sents[i].rstrip().endswith('?'):
            return i
    return len(sents) - 1


@dataclass
class Rendering:
    """One way of presenting the same problem."""
    name: str
    text: str
    ok: bool = True
    refused: str = ''
    meta: Dict[str, object] = field(default_factory=dict)


def is_faithful(original: str, rendered: str) -> Tuple[bool, str]:
    """Does `rendered` still state the same problem?

    Checked, in order: it is non-empty prose, it still asks something, it
    carries exactly the same multiset of numeric literals, and it has not
    collapsed to a fraction of the original's length (which is how a model
    "paraphrases" by answering instead).
    """
    r = (rendered or '').strip()
    if len(r) < 20:
        return False, 'rendering is empty or truncated'
    if '?' in original and '?' not in r:
        return False, 'the question was dropped'
    a, b = numbers(original), numbers(r)
    if a != b:
        missing = {k: v for k, v in a.items() if b.get(k, 0) != v}
        extra = {k: v for k, v in b.items() if a.get(k, 0) != v}
        return False, f'numbers changed (orig {missing} -> rendered {extra})'
    if len(r) < 0.5 * len(original.strip()):
        return False, 'rendering lost half the text'
    if len(r) > 3.0 * len(original.strip()):
        return False, 'rendering tripled the text'
    return True, ''


# ---------------------------------------------------------------------------
# structural transforms -- pure re-layout, faithful by construction
# ---------------------------------------------------------------------------

def _label(i: int) -> str:
    """(a) (b) (c) ... and a plain dash past the alphabet. Never a digit."""
    return f'({chr(97 + i)})' if i < 26 else '-'


def identity(text: str) -> Rendering:
    """The problem as given. This is also the control arm's input, so its
    answer is reused rather than paid for twice."""
    return Rendering('identity', text.strip())


def givens_first(text: str) -> Rendering:
    """Facts enumerated, then the question.

    Sentences are moved as whole units and never edited, so the numeric
    multiset is preserved by construction. Refused when there is nothing to
    restructure -- a one-sentence problem has no layout to change, and
    returning the identity under a different name would silently collapse the
    ensemble to a single voter.
    """
    sents = split_sentences(text)
    if len(sents) < 3:
        return Rendering('givens_first', text.strip(), ok=False,
                         refused=f'only {len(sents)} sentence(s) to restructure')
    q = _question_index(sents)
    facts = [s for i, s in enumerate(sents) if i != q]
    # Labels are letters, not digits. An enumerated "1. 2. 3." reads better and
    # is wrong: those digits are numeric literals, `is_faithful` counts them,
    # and the rendering refuses itself. Caught by test_v20 PART 1 -- which is
    # the check working, not a nuisance: a transform that adds a quantity to a
    # word problem is exactly the defect this module exists to refuse.
    body = '\n'.join(f'{_label(i)} {s}' for i, s in enumerate(facts))
    out = f'Known facts:\n{body}\n\nQuestion: {sents[q]}'
    ok, why = is_faithful(text, out)
    return Rendering('givens_first', out, ok=ok, refused=why,
                     meta={'n_facts': len(facts)})


def goal_first(text: str) -> Rendering:
    """The question first, then the facts in their original order.

    Goal-first framing is one of the standard meaning-preserving families in
    the metamorphic-testing literature. Facts keep their narrative order, so
    unlike a reordering transform this is safe on problems whose sentences
    depend on each other ("Then Mark ate ...").
    """
    sents = split_sentences(text)
    if len(sents) < 3:
        return Rendering('goal_first', text.strip(), ok=False,
                         refused=f'only {len(sents)} sentence(s) to restructure')
    q = _question_index(sents)
    facts = [s for i, s in enumerate(sents) if i != q]
    out = (f'Find: {sents[q]}\n\nYou are given:\n'
           + '\n'.join(f'- {s}' for s in facts))
    ok, why = is_faithful(text, out)
    return Rendering('goal_first', out, ok=ok, refused=why,
                     meta={'n_facts': len(facts)})


# ---------------------------------------------------------------------------
# the one transform that goes through the model
# ---------------------------------------------------------------------------

PARAPHRASE_PROMPT = """Restate the following word problem in different words.

Rules, all of them mandatory:
- Keep every number exactly as it appears, in digits. Do not round, do not \
spell a number out, do not add a number, do not drop one.
- Keep the question being asked identical in meaning, and keep it as a question.
- Change the wording, the sentence structure and the order of the narrative \
where you can do so without changing what is being asked.
- Do not solve the problem. Do not add hints, steps or commentary.

Output the restated problem between the markers and nothing else:

<restated>
...the restated problem...
</restated>

Problem:
{problem}"""

_RESTATED = re.compile(r'<restated>(.*?)</restated>', re.S | re.I)


def paraphrase_prompt(text: str) -> str:
    return PARAPHRASE_PROMPT.format(problem=text.strip())


def parse_paraphrase(raw: str, original: str) -> Rendering:
    """Pull the restated problem out of a model response and validate it.

    An unfaithful paraphrase is refused, never patched. This is the rule v16.2
    and v19 were missing: a construction that changed the problem was allowed
    downstream and the whole arm then measured the construction's failure
    instead of the mechanism's.
    """
    raw = str(raw or '')
    m = _RESTATED.search(raw)
    if not m:
        # Keep a slice of what the model actually said. A refusal rate is a
        # number; a refusal you cannot read is a dead end, and the smoke test
        # hit 2/2 with no way to see why without paying for another run.
        return Rendering('paraphrase', original.strip(), ok=False,
                         refused='no <restated> block in the response',
                         meta={'raw': raw[:400]})
    out = m.group(1).strip()
    ok, why = is_faithful(original, out)
    return Rendering('paraphrase', out if ok else original.strip(),
                     ok=ok, refused=why,
                     meta={} if ok else {'raw': out[:400]})


# ---------------------------------------------------------------------------
# the set under test
# ---------------------------------------------------------------------------

STRUCTURAL = {'identity': identity, 'givens_first': givens_first,
              'goal_first': goal_first}

#: The pre-registered rendering set for the v20 pre-test. All three are
#: STRUCTURAL, which means zero presenter calls: `identity` is free (the
#: control already paid for it) and the other two cost one solver call each.
#:
#: [AMENDED after the 2026-09-22 smoke test, before the real run.] The set was
#: `(identity, givens_first, paraphrase)` and the paraphrase was refused on
#: 2 of 2 rows. That is not a tolerable failure rate, it is a collapse: with
#: the paraphrase gone the ensemble had two voters, every row was a 1-1 tie,
#: every tie broke to the identity, and arm F became the control with a
#: presenter call attached -- W=0 L=0 against C, by construction rather than by
#: measurement. Qwen2.5-MATH is tuned to solve, not to restate, so asking it
#: for a faithful re-rendering is asking the wrong instrument.
#:
#: `goal_first` replaces it: free, deterministic, faithful by construction and
#: usable on 92% of gsm-symbolic-p2. The paraphrase is still selectable with
#: --renderings and is worth revisiting with a non-math model as the presenter,
#: because it is the transform the prior work uses.
DEFAULT_SET = ('identity', 'givens_first', 'goal_first')


def build_structural(text: str, names) -> List[Rendering]:
    """Every structural rendering named in `names`, in order. Refusals are
    returned rather than dropped, so the caller can record how often the
    construction -- not the model -- was the limit."""
    return [STRUCTURAL[n](text) for n in names if n in STRUCTURAL]


def agree(a: Optional[float], b: Optional[float]) -> bool:
    """The repo's numeric tolerance, used here to decide whether two
    renderings answered the same."""
    if a is None or b is None:
        return False
    return abs(a - b) <= max(1e-3, 1e-4 * abs(b))


def majority(votes: List[Tuple[str, Optional[float]]],
             default_name: str = 'identity') -> Tuple[Optional[float], Dict]:
    """Majority answer over renderings, ties broken by `default_name`.

    Ties break toward the identity rendering because that is the deployable
    default: on a tie the system has learned nothing and must fall back to what
    a single-call system would have said. Breaking ties any other way would
    quietly credit the ensemble for coin flips, which is the bug the v12
    selection layer shipped twice.
    """
    live = [(n, v) for n, v in votes if v is not None]
    info: Dict[str, object] = {'n_votes': len(live), 'tie': False,
                               'unanimous': False}
    if not live:
        return None, info
    clusters: List[List[Tuple[str, float]]] = []
    for name, v in live:
        for c in clusters:
            if agree(v, c[0][1]):
                c.append((name, v))
                break
        else:
            clusters.append([(name, v)])
    clusters.sort(key=len, reverse=True)
    info['unanimous'] = len(clusters) == 1 and len(live) > 1
    info['cluster_sizes'] = [len(c) for c in clusters]
    top = clusters[0]
    if len(clusters) > 1 and len(clusters[1]) == len(top):
        info['tie'] = True
        for name, v in live:
            if name == default_name:
                return v, info
    return top[0][1], info


def disagreement_rate(values: List[Optional[float]]) -> Optional[float]:
    """Fraction of unordered pairs that disagree.

    This is the measurement the accuracy result depends on and the one that
    survives a null: if the renderings disagree no more often than temperature
    samples do, then input-space diversity is not producing diversity and the
    mechanism is dead regardless of which arm scored higher.
    """
    live = [v for v in values if v is not None]
    if len(live) < 2:
        return None
    pairs = dis = 0
    for i in range(len(live)):
        for j in range(i + 1, len(live)):
            pairs += 1
            if not agree(live[i], live[j]):
                dis += 1
    return dis / pairs if pairs else None
