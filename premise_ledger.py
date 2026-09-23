"""
[v21.0] Premise ledger: which stated quantities did a derivation actually spend?

WHY
---
SIV's own abstract states its blind spot: it audits math -> math, and "cannot
detect errors in the NL -> math translation layer". The commonest translation
error a 7B model makes on GSM-Symbolic-P2 is not a wrong operation but a
MISSING one: a clause the solution never touches. Reading the 24 rows that
self-consistency@3 got wrong in pretest_v20.json, the majority answer is the
answer you get by deleting one clause on, among others,

    p2_389   1800 lb / 120 lb per trip = 15   (the "three trucks" never used)
    p2_2185  32 / 96 = 33.3%                  (the "two times as many wasps")
    p2_2184  30 / 120 = 25%                   (the "four times as many ants")
    p2_1291  33 + 27 = 60                     (the "3 cheerleaders" never used)

and voting cannot fix this, because every sample drops the same clause: the
error is correlated, which is exactly the regime where majority voting "locks
in" the wrong answer (Choi et al., Debate or Vote, 2025).

A clause that was never used leaves a fingerprint that needs no model to read:
one of its quantities never appears as an operand. This module reads it.

WHAT
----
    ledger = extract_premises(problem_text)       # quantities the text states
    ops    = cot_operands(raw) or code_operands(src)  # what a solution operates on
    audit(ledger, ops) -> Audit(consumed, unconsumed)

Deterministic, no model, no gold. It is a DETECTOR of omissions only: a
solution that uses every quantity can still be wrong, and a quantity can be
legitimately irrelevant (GSM-NoOp distractors, "7 feet wide" in a spacing
problem). So nothing here decides an answer on its own; see premise_vote().

Matching is deliberately generous about HOW a quantity was spent (percent as
0.2 or 0.8, minutes as hours, a factor folded into one product) because a
false "unconsumed" flag is the expensive error: it would demote a correct
majority. A missed omission only costs the gain.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

# ---------------------------------------------------------------------------
# premises
# ---------------------------------------------------------------------------

_UNITS = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
    'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12,
    'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16,
    'seventeen': 17, 'eighteen': 18, 'nineteen': 19,
}
_TENS = {'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60,
         'seventy': 70, 'eighty': 80, 'ninety': 90}
_DENOMS = {'half': 2, 'halves': 2, 'third': 3, 'thirds': 3, 'quarter': 4,
           'quarters': 4, 'fourth': 4, 'fourths': 4, 'fifth': 5, 'fifths': 5,
           'sixth': 6, 'sixths': 6, 'seventh': 7, 'sevenths': 7, 'eighth': 8,
           'eighths': 8, 'ninth': 9, 'ninths': 9, 'tenth': 10, 'tenths': 10}
_MULTIPLIERS = {'twice': 2, 'double': 2, 'doubled': 2, 'doubles': 2,
                'thrice': 3, 'triple': 3, 'tripled': 3, 'triples': 3,
                'quadruple': 4, 'quadrupled': 4, 'dozen': 12, 'dozens': 12}

_WORD = re.compile(r"[A-Za-z]+")
# digits, optional thousands separators and decimal tail; no word boundary on
# the right because GSM-Symbolic writes "7feet" and "5-pound".
_DIGITS = re.compile(r"(?<![\d.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?")
_FRACTION = re.compile(r"(?<![\d.])(\d+)\s*/\s*(\d+)(?![\d.])")
_PERCENT_AFTER = re.compile(r"^\s*(%|percent\b|per\s*cent\b)", re.I)


@dataclass(frozen=True)
class Premise:
    """One quantity the problem states."""
    value: float
    surface: str                  # the text as written ("two times", "20%")
    sentence: int                 # index into split_sentences(text)
    forms: FrozenSet[float]       # values under which spending it counts
    kind: str = 'number'          # number | percent | fraction | word
    also: Tuple[int, ...] = ()    # other sentences restating the same quantity

    def short(self) -> str:
        return f"{self.surface!r}@s{self.sentence}"

    @property
    def sentences(self) -> Tuple[int, ...]:
        return (self.sentence,) + self.also


# Unit rescalings a derivation may legitimately apply before writing a
# quantity down: percent, minutes/hours, hours/days, days/weeks, months/year,
# dozens, thousands. Deliberately generous -- see the module docstring.
_SCALES = (1.0, 100.0, 60.0, 24.0, 7.0, 12.0, 1000.0)


def _forms(v: float, kind: str) -> FrozenSet[float]:
    """Values under which spending a premise counts.

    Unit rescalings apply to counts and measures only. A fraction or a percent
    rescaled by 24 or 60 is not a unit change, it is a collision: "half"
    would claim every 12 in the solution (half a day in hours).
    """
    out: Set[float] = {v}
    if kind == 'percent':
        p = v / 100.0
        out |= {p, 1 - p, 1 + p, 100 - v, 100 + v}
    elif kind == 'fraction':
        out |= {100 * v}
        if 0 < v < 1:
            out |= {1 - v, 1 / v}
    elif v != 1:
        # a stated 1 rescaled is every unit constant at once (1 h = 60,
        # 1 wk = 7, 1 dozen = 12); letting it claim those would steal the
        # slots of the premises that really are 60, 7 or 12.
        for sc in _SCALES:
            out.add(v * sc)
            out.add(v / sc)
    return frozenset(round(x, 9) for x in out if x == x)


def _parts(*xs: float) -> FrozenSet[float]:
    """Numerator and denominator of a fraction, spendable as written."""
    return frozenset(round(float(x), 9) for x in xs)


def split_sentences(text: str) -> List[str]:
    """Same splitter as renderings.py, so sentence indices line up."""
    import renderings as R
    return R.split_sentences(text)


def _word_numbers(sent: str) -> List[Tuple[float, str, str]]:
    """(value, surface, kind) for spelled-out quantities in one sentence."""
    toks = [(m.group(0).lower(), m.start(), m.end()) for m in _WORD.finditer(sent)]
    out: List[Tuple[float, str, str]] = []
    i = 0
    while i < len(toks):
        w, s, e = toks[i]
        # "two-thirds", "three sixths", "a third of". Without a numerator word
        # "third" is an ordinal ("on the third day") and is not a quantity.
        if (w in _UNITS or w == 'a') and i + 1 < len(toks) and toks[i + 1][0] in _DENOMS:
            num = 1 if w == 'a' else _UNITS[w]
            den = _DENOMS[toks[i + 1][0]]
            out.append((num / den, sent[s:toks[i + 1][2]], 'fraction:%d/%d' % (num, den)))
            i += 2
            continue
        if w in ('half', 'halves'):
            out.append((0.5, sent[s:e], 'fraction'))
            i += 1
            continue
        if w in ('quarter', 'quarters') and i + 1 < len(toks) and toks[i + 1][0] == 'of':
            out.append((0.25, sent[s:e], 'fraction'))
            i += 1
            continue
        if w in _MULTIPLIERS:
            out.append((float(_MULTIPLIERS[w]), sent[s:e], 'word'))
            i += 1
            continue
        if w in _TENS:
            v = _TENS[w]
            j = i + 1
            if j < len(toks) and toks[j][0] in _UNITS and 0 < _UNITS[toks[j][0]] < 10:
                v += _UNITS[toks[j][0]]
                j += 1
            end = toks[j - 1][2]
            if j < len(toks) and toks[j][0] in ('hundred', 'thousand'):
                v *= 100 if toks[j][0] == 'hundred' else 1000
                end = toks[j][2]
                j += 1
            out.append((float(v), sent[s:end], 'word'))
            i = j
            continue
        if w in _UNITS and w != 'zero':
            v = _UNITS[w]
            j = i + 1
            end = e
            if j < len(toks) and toks[j][0] in ('hundred', 'thousand'):
                v *= 100 if toks[j][0] == 'hundred' else 1000
                end = toks[j][2]
                j += 1
            # "one" is a pronoun more often than a quantity ("the red one",
            # "one of the boxes"). Dropping a premise can only make the
            # detector MISS an omission, never raise a false flag.
            if v != 1:
                out.append((float(v), sent[s:end], 'word'))
            i = j
            continue
        i += 1
    return out


def extract_premises(text: str, merge_repeats: bool = True) -> List[Premise]:
    """Every quantity the problem text states, in order, with its sentence.

    With `merge_repeats`, a value stated the same way more than once ("in the
    first 16 seconds ... in the next 16 seconds ...") is ONE premise that one
    operand pays for. Otherwise a solution that counts popcorn without ever
    computing a per-second rate is "missing" four premises that never mattered,
    and one that does compute a rate is rewarded for style.
    """
    prem = _extract_all(text)
    if not merge_repeats:
        return prem
    merged: Dict[Tuple[float, str], int] = {}
    out: List[Premise] = []
    for p in prem:
        key = (round(p.value, 9), p.kind)
        if key in merged:
            k = merged[key]
            q = out[k]
            if p.sentence != q.sentence and p.sentence not in q.also:
                out[k] = Premise(q.value, q.surface, q.sentence, q.forms, q.kind,
                                 q.also + (p.sentence,))
            continue
        merged[key] = len(out)
        out.append(p)
    return out


def _extract_all(text: str) -> List[Premise]:
    prem: List[Premise] = []
    for si, sent in enumerate(split_sentences(text)):
        taken: List[Tuple[int, int]] = []
        for m in _FRACTION.finditer(sent):
            a, b = int(m.group(1)), int(m.group(2))
            if b == 0:
                continue
            taken.append((m.start(), m.end()))
            v = a / b
            prem.append(Premise(v, m.group(0), si,
                                _forms(v, 'fraction') | _parts(a, b), 'fraction'))
        for m in _DIGITS.finditer(sent):
            if any(s <= m.start() < e for s, e in taken):
                continue
            raw = m.group(1).replace(',', '') + ('.' + m.group(2) if m.group(2) else '')
            v = float(raw)
            kind = 'percent' if _PERCENT_AFTER.match(sent[m.end():]) else 'number'
            prem.append(Premise(v, m.group(0), si, _forms(v, kind), kind))
        for v, surf, kind in _word_numbers(sent):
            if kind.startswith('fraction'):
                # "two-thirds" is spent as 2/3, as 0.667, or as "* 2 / 3"
                f = _forms(v, 'fraction')
                if ':' in kind:
                    num, den = kind.split(':')[1].split('/')
                    f = f | _parts(num, den)
                kind = 'fraction'
            else:
                f = _forms(v, kind)
            prem.append(Premise(v, surf, si, f, kind))
    return prem


# ---------------------------------------------------------------------------
# operands
# ---------------------------------------------------------------------------

_MATH_BLOCK = re.compile(r"\\\[(.*?)\\\]|\\\((.*?)\\\)|\$\$(.*?)\$\$|\$(.*?)\$", re.S)
_FRAC_TEX = re.compile(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
_NUM = re.compile(r"(?<![A-Za-z_])-?\d+(?:,\d{3})*(?:\.\d+)?")
_OP_BETWEEN_NUMBERS = re.compile(r"\d\s*(?:[=×÷*/+\-]|\\times|\\cdot|\\div)\s*\(?\d")
_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_RESULT = re.compile(r"=\s*(-?\d[\d,]*(?:\.\d+)?)")


def _nums(s: str) -> List[float]:
    out = []
    for t in _NUM.findall(s):
        try:
            out.append(abs(float(t.replace(',', ''))))
        except ValueError:
            pass
    return out


class Operands(list):
    """Operand occurrences of one solution: (value, chunk) pairs.

    The chunk is the equation or code line the number sits in. It is what lets
    the audit say WHICH premise was dropped when two premises share a value:
    "\frac{64}{2}" spends the "half" that sits next to 64 in the text, not the
    "two times as many wasps" three clauses earlier.
    """

    def __init__(self, *a):
        super().__init__(*a)
        # values some written equation PRODUCES ("= 32"). They are explained
        # by that equation's own operands, so they cannot be evidence that a
        # different premise was spent by folded arithmetic.
        self.results: Set[float] = set()

    def counter(self) -> Counter:
        return Counter(v for v, _ in self)


def _chunk_values(c: str) -> List[float]:
    vals: List[float] = []
    c = _BOXED.sub(r" \1 ", c)
    for fm in _FRAC_TEX.finditer(c):
        a, b = _nums(fm.group(1)), _nums(fm.group(2))
        vals.extend(a + b)
        if len(a) == 1 and len(b) == 1 and b[0]:
            vals.append(a[0] / b[0])
    c = _FRAC_TEX.sub(' ', c)
    c = re.sub(r"\%|%", ' ', c)
    vals.extend(_nums(c))
    return [round(x, 9) for x in vals]


def cot_operands(raw: str) -> Operands:
    """Numbers a free-form CoT OPERATES on, as a multiset.

    Prose restatement is not spending ("Oscar has 96 puppies" costs nothing),
    so only math is read: LaTeX blocks, plus any plain line with an operator
    between two numbers. \frac{a}{b} contributes a, b and a/b.

    A multiset, because small values collide: "two times as many wasps" and
    "half of which" both surface as a 2, and a solution that divides by 2 once
    has spent ONE of them, not both. Identical chunks count once, since a model
    that restates "64 / 2 = 32" in its summary has not halved twice.
    """
    out = Operands()
    if not raw:
        return out
    chunks: List[str] = []
    for m in _MATH_BLOCK.finditer(raw):
        chunks.append(next(g for g in m.groups() if g is not None))
    body = _MATH_BLOCK.sub(' ', raw)
    for line in body.splitlines():
        if _OP_BETWEEN_NUMBERS.search(line):
            chunks.append(line)
    seen: Set[str] = set()
    for ci, c in enumerate(chunks):
        key = re.sub(r"\s+", '', c)
        if not key or key in seen:
            continue
        seen.add(key)
        out.extend((v, ci) for v in _chunk_values(c))
        for m in _RESULT.finditer(c):
            out.results.update(_chunk_values(m.group(1)))
    return out


def code_operands(src: str) -> Operands:
    """Numeric literals in a program, comments and strings removed; one chunk
    per line."""
    out = Operands()
    if not src:
        return out
    for li, line in enumerate(src.splitlines()):
        line = re.sub(r"#.*$", '', line)
        line = re.sub(r"(['\"]).*?\1", ' ', line)
        out.extend((round(x, 9), li) for x in _nums(line))
    return out


def _as_operands(ops) -> Operands:
    if isinstance(ops, Operands):
        return ops
    if isinstance(ops, Counter):
        o = Operands()
        for v, n in sorted(ops.items()):
            o.extend([(v, -1)] * n)
        return o
    return Operands(ops)


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------

def _close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-6 * max(1.0, abs(a), abs(b))


def _direct(p: Premise, slots: Operands) -> List[int]:
    return [k for k, (o, _) in enumerate(slots) if any(_close(f, o) for f in p.forms)]


def _one_hop(p: Premise, i: int, vals: List[float], slots: Operands,
             free: Set[int]) -> List[int]:
    """Free operands equal to p combined with another premise in one step.

    Not in a chunk that writes p's value out explicitly: there the explicit
    operand is how p would have been spent, and if a different premise owns
    it, p was not spent there. Nor on a value some equation PRODUCES: without
    that, "\frac{64}{2} = 32" pays for "half of which" through its 2 AND for
    "two times as many wasps" through its 32 (= 64 / 2) -- one halving,
    counted as two premises.
    """
    derived: Set[float] = set()
    for j, q in enumerate(vals):
        if j == i:
            continue
        derived |= {p.value + q, abs(p.value - q), p.value * q}
        if q:
            derived.add(p.value / q)
        if p.value:
            derived.add(q / p.value)
    explicit = {c for (o, c) in slots if c >= 0 and any(_close(f, o) for f in p.forms)}
    produced = getattr(slots, 'results', set())
    return [k for k in sorted(free)
            if slots[k][1] not in explicit
            and not any(_close(slots[k][0], r) for r in produced)
            and any(_close(d, slots[k][0]) for d in derived)]


def _match(order: Sequence[int], cand: Dict[int, List[int]],
           owner: Dict[int, int]) -> None:
    """Augmenting-path bipartite matching, in place on `owner` (slot -> premise).
    Only premises in `order` can be displaced along a path."""
    movable = set(order)

    def augment(i: int, seen: Set[int]) -> bool:
        for k in cand[i]:
            if k in seen:
                continue
            seen.add(k)
            if k not in owner or (owner[k] in movable and augment(owner[k], seen)):
                owner[k] = i
                return True
        return False

    for i in order:
        augment(i, set())


def _affinity(i: int, k: int, premises: Sequence[Premise], slots: Operands) -> int:
    """How many OTHER numbers in slot k's chunk come from premise i's sentence.
    A chunk that also holds 64 is where the "half" of "64 ..., half of which"
    was spent."""
    chunk = slots[k][1]
    if chunk < 0:
        return 0
    same = [q for j, q in enumerate(premises)
            if j != i and q.sentence == premises[i].sentence]
    n = 0
    for kk, (o, c) in enumerate(slots):
        if kk != k and c == chunk and any(_close(f, o) for q in same for f in q.forms):
            n += 1
    return n


@dataclass
class Audit:
    consumed: List[Premise] = field(default_factory=list)
    unconsumed: List[Premise] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.unconsumed

    @property
    def n_unconsumed(self) -> int:
        return len(self.unconsumed)

    @property
    def coverage(self) -> float:
        n = len(self.consumed) + len(self.unconsumed)
        return 1.0 if n == 0 else len(self.consumed) / n


def audit(premises: Sequence[Premise], ops, one_hop: bool = True) -> Audit:
    """Which premises a solution spent: a maximum matching between premises
    and operand occurrences, each occurrence spendable once.

    A premise is matched by an operand equal to one of its forms, or -- with
    `one_hop`, and only on an operand no premise claimed directly -- by an
    operand equal to it combined with ANOTHER premise in one arithmetic step. The second rule
    keeps a model that writes "wasps = 24" instead of "2 x 12 = 24" from being
    flagged: it folded the 2 into a product whose other factor the text states.

    The vote uses the COUNT of unconsumed premises. WHICH premise is left over is decided, when values collide, by
    chunk affinity (see _affinity); that is a best guess, used only to word a
    reminder.
    """
    slots = _as_operands(ops)
    vals = [p.value for p in premises]
    idx = range(len(premises))

    def ranked(i: int, ks: List[int]) -> List[int]:
        return sorted(ks, key=lambda k: -_affinity(i, k, premises, slots))

    # Phase 1: a quantity written down as itself claims its operand first.
    cand = {i: ranked(i, _direct(premises[i], slots)) for i in idx}
    aff = {i: (_affinity(i, cand[i][0], premises, slots) if cand[i] else 0) for i in idx}
    owner: Dict[int, int] = {}
    _match(sorted(idx, key=lambda i: (-aff[i], len(cand[i]), i)), cand, owner)

    # Phase 2: folded-arithmetic credit, only on operands nobody claimed
    # directly. Letting it compete in phase 1 made a stated 17 that was never
    # used (|17 - 12| = 5) steal the slot of a stated 5, and the reminder then
    # quoted the wrong sentence.
    if one_hop:
        done = set(owner.values())
        free = set(range(len(slots))) - set(owner)
        rest = [i for i in idx if i not in done]
        cand2 = {i: ranked(i, _one_hop(premises[i], i, vals, slots, free)) for i in rest}
        _match(sorted(rest, key=lambda i: (len(cand2[i]), i)), cand2, owner)
    matched = set(owner.values())
    a = Audit()
    for i, p in enumerate(premises):
        (a.consumed if i in matched else a.unconsumed).append(p)
    return a


# ---------------------------------------------------------------------------
# the vote
# ---------------------------------------------------------------------------

def _agree(a: Optional[float], b: Optional[float]) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= max(1e-3, 1e-4 * abs(b))


def clusters(answers: Sequence[Optional[float]]) -> List[List[int]]:
    """Indices grouped by agreeing answer, largest first, ties by first seen."""
    groups: List[List[int]] = []
    for i, a in enumerate(answers):
        if a is None:
            continue
        for g in groups:
            if _agree(answers[g[0]], a):
                g.append(i)
                break
        else:
            groups.append([i])
    groups.sort(key=lambda g: (-len(g), g[0]))
    return groups


def premise_vote(answers: Sequence[Optional[float]],
                 audits: Sequence[Audit]) -> Tuple[Optional[float], Dict]:
    """Majority vote in which a derivation that spent LESS of the problem than
    another derivation in the same pool does not count.

    Rule (pre-registered, one line): restrict the vote to the samples with the
    fewest unconsumed premises, then take the plurality among them; a tie among
    them goes to the plain-majority answer if it is one of the tied ones.

    If every sample leaves the same number of quantities unused (a shared
    distractor, or a shared omission) nobody is demoted and this IS plain
    majority voting. The rule can only move the answer when the pool itself
    shows that more of the problem was spendable.
    """
    idx = [i for i, a in enumerate(answers) if a is not None]
    if not idx:
        return None, {'rule': 'empty', 'moved': False, 'eligible': [], 'demoted': []}
    best = min(audits[i].n_unconsumed for i in idx)
    eligible = [i for i in idx if audits[i].n_unconsumed == best]
    plain = clusters(answers)
    sub = clusters([answers[i] if i in eligible else None
                    for i in range(len(answers))])
    top = [g for g in sub if len(g) == len(sub[0])]
    pick_group = next((g for g in top if _agree(answers[g[0]], answers[plain[0][0]])),
                      top[0])
    pick = answers[pick_group[0]]
    return pick, {'rule': 'fewest-unconsumed', 'eligible': eligible,
                  'demoted': [i for i in idx if i not in eligible],
                  'moved': not _agree(pick, answers[plain[0][0]])}


def shared_omissions(premises: Sequence[Premise],
                     audits: Sequence[Audit]) -> List[Premise]:
    """Premises left unconsumed by EVERY derivation in the pool, text order.

    Identity is by position in `premises`, so this is only as exact as the
    matching; it is used to write a reminder, never to score.
    """
    if not audits:
        return []
    common = set.intersection(*({id(p) for p in a.unconsumed} for a in audits))
    return [p for p in premises if id(p) in common]
