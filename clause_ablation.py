"""
[v39.0] Clause ablation: a test-time detector of the sentences a model IGNORES.

WHY THIS VERSION EXISTS
-----------------------
On GSM-Symbolic P2 the residual errors are mostly readings, not arithmetic:
12 of the 18 hand-classified rows with no right sample apply the familiar GSM8K
solution and ignore a clause that P2 ADDED (RoR-Bench calls it "recitation"
and reports no satisfactory remedy). Every same-family check tried so far
confirms that reading (the PRM scores 7 of 8 such errors 1.000; a Qwen Reader
recites too), and voting cannot see it when the samples agree. v24 showed that
an intervention on the READING helps where the model misread (+19.6 pp) and
hurts where it read correctly (-12.5 pp): what is missing is a detector that
says WHETHER and WHERE the model misread, without a gold answer.

THE DETECTOR (this file is pure Python; the generations are pretest_v39.py)
--------------------------------------------------------------------------
Split the problem into units: every non-question sentence, every parenthetical,
and the leading (or trailing "if ...") clause of the question. Delete one unit
at a time and solve each variant greedily. If deleting an INFORMATIVE unit (one
that holds a number or a condition word) leaves the answer unchanged, the
answer does not depend on that unit: the model ignored it. The same-family
self-consistency that defeated v27-v32 (the model confirms its own reading) is
here the signal, not the obstacle: a model that recites ignores the clause in
every variant, so the variant without it gives the same answer.

WHAT IS AND IS NOT NEW (checked 2026-10-21)
------------------------------------------
Deleting a condition is a known EVALUATION probe: CogMath (arXiv 2506.04481)
removes a key condition and reads an unchanged answer as memorisation, TreeCut
(arXiv 2502.13442) and MiP-GSM8K (arXiv 2504.06514) build unanswerable
problems, "Pause or Fabricate?" (arXiv 2604.19656) masks premises to label
training data. ContextCite (arXiv 2409.00729) attributes a generation to
context sentences by ablation. GSM-IC and I3C (arXiv 2403.12744) target
irrelevant conditions. Not found: answer invariance under deletion used per
problem, at test time and without labels, to flag a wrong consensus and point
at the condition it ignored, as the gate of a targeted repair.

The pilot (pretest_v39.py, V39_PLAN.md) measures the detector only: does it
fire more on confidently-wrong than on confidently-right rows, and do the
units it flags coincide with the clauses P2 added? The repair comes after.
"""
from __future__ import annotations

import re
from math import comb
from typing import Dict, List, Optional, Sequence, Tuple

import renderings as RD

# ---------------------------------------------------------------- frozen constants (V39_PLAN.md section 3)
ADDED_BELOW = 0.5        # best containment in any reference sentence below this: the unit is ADDED
SAME_FROM = 0.75         # at or above this: the unit is unchanged (between: MODIFIED)
CONST_SHARE = 0.9        # a word in at least this share of a template's instances is template text
INVARIANT_TOL = 1e-3     # two answers are the same answer: the project's canonical grader
PRIMARY_KINDS = ('sentence', 'paren')   # the pre-registered flag deletes sentences and parentheticals

_NUMBER_WORDS = set(('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen '
                     'fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty '
                     'ninety hundred thousand million dozen half halves twice double triple thrice third thirds '
                     'quarter quarters fifth fifths percent').split())
_CUES = set(('if when once after before each every per except including include includes only all both not no '
             'none without remaining rest return returns returned again back more less fewer than extra another '
             'also however but instead total entire whole same').split())
_STOP = set(('a an the and or of to in on at for with by from as is are was were be been being it its this that '
             'these those they them their there he she his her him we our you your i my me do does did has have '
             'had will would can could should shall may might must so then also some any how what which who whom '
             'whose why where many much into out up down over under about just one').split())
_QWORDS = r'(?:how|what|which|when|where|who|whom|whose|why|calculate|find|determine|compute)'
_PAREN = re.compile(r'\s*\([^()]{3,}\)')
_QLEAD = re.compile(r'^(?P<clause>.{8,}?),\s+(?=' + _QWORDS + r'\b)', re.I)
_QTAIL = re.compile(r',?\s+(?P<clause>if\s[^?]{4,})(?=\?\s*$)', re.I)
_WORD = re.compile(r"[A-Za-z]+")


# ---------------------------------------------------------------- units
def units(text: str) -> Tuple[List[str], int, List[Dict]]:
    """(sentences, index of the question sentence, units). A unit is a dict with
    kind ('sentence' | 'paren' | 'qclause'), the sentence index, its text and,
    for sub-sentence units, the character span inside that sentence."""
    sents = RD.split_sentences(text)
    qi = RD._question_index(sents)
    out: List[Dict] = []
    for i, s in enumerate(sents):
        if i != qi:
            out.append({'kind': 'sentence', 'sent': i, 'text': s})
        for m in _PAREN.finditer(s):
            out.append({'kind': 'paren', 'sent': i, 'text': m.group(0).strip(), 'span': [m.start(), m.end()]})
        if i == qi:
            m = _QLEAD.match(s)
            if m:
                out.append({'kind': 'qclause', 'sent': i, 'text': m.group('clause'), 'span': [0, m.end()]})
            t = _QTAIL.search(s)
            if t:
                out.append({'kind': 'qclause', 'sent': i, 'text': t.group('clause'), 'span': [t.start(), t.end()]})
    return sents, qi, out


def delete(sents: Sequence[str], unit: Dict) -> str:
    """The problem text without `unit`."""
    out = list(sents)
    if unit['kind'] == 'sentence':
        out[unit['sent']] = ''
    else:
        s = sents[unit['sent']]
        a, b = unit['span']
        t = s[:a] + s[b:]
        t = re.sub(r'\s{2,}', ' ', t).strip()
        t = re.sub(r'\s+([,.?!;:])', r'\1', t)
        if a == 0 and t:
            t = t[0].upper() + t[1:]
        out[unit['sent']] = t
    return ' '.join(x for x in out if x)


def informative(text: str) -> bool:
    """Holds a number (digits or a number word) or a condition word."""
    if re.search(r'\d', text):
        return True
    words = {w.lower() for w in _WORD.findall(text)}
    return bool(words & _NUMBER_WORDS) or bool(words & _CUES)


# ---------------------------------------------------------------- which units did P2 add? (analysis labels only)
# A GSM-Symbolic template is instantiated 50 times with different names, places,
# items and numbers. The words that occur in (nearly) every instance are the
# template's own text; the rest are slot fills. Comparing only template words
# against the base template (GSM-Symbolic 'main', same id) separates the clauses
# P2 added from the ones it kept, whatever the slots were filled with. GSM-Plus
# keeps the seed problem's wording, so there the seed question is compared as is.
def content(text: str) -> set:
    """Lower-cased content words, stop words and short words out, a plural 's' folded."""
    out = set()
    for t in (w.lower() for w in _WORD.findall(text)):
        if t in _STOP or len(t) <= 2:
            continue
        out.add(t[:-1] if len(t) > 3 and t.endswith('s') else t)
    return out


def template_words(instances: Sequence[str]) -> set:
    """Content words present in at least CONST_SHARE of a template's instances."""
    seen: Dict[str, int] = {}
    for t in instances:
        for w in content(t):
            seen[w] = seen.get(w, 0) + 1
    return {w for w, n in seen.items() if n >= CONST_SHARE * len(instances)}


def containment(words: set, reference: Sequence[set]) -> float:
    """Best share of `words` found inside one reference sentence (1.0 for an empty unit)."""
    if not words:
        return 1.0
    return max([len(words & s) / len(words) for s in reference] or [0.0])


def label_of(c: float) -> str:
    return 'added' if c < ADDED_BELOW else 'same' if c >= SAME_FROM else 'modified'


def annotate(text: str, reference: Optional[Sequence[str]] = None, keep: Optional[set] = None,
             ref_keep: Optional[set] = None) -> Tuple[List[str], int, List[Dict]]:
    """units() plus, per unit, `informative` and, when the reference problem is
    known (its sentences), the added/modified/same label. `keep` restricts the
    unit's words to template words, `ref_keep` the reference's (GSM-Symbolic);
    both None compares plain content words (GSM-Plus)."""
    sents, qi, us = units(text)
    ref = None
    if reference:
        ref = [content(s) & ref_keep if ref_keep is not None else content(s) for s in reference]
    for u in us:
        u['informative'] = informative(u['text'])
        if ref is not None:
            w = content(u['text'])
            if keep is not None:
                w &= keep
            u['containment'] = round(containment(w, ref), 3)
            u['label'] = label_of(u['containment'])
    return sents, qi, us


# ---------------------------------------------------------------- the textual baseline (no generation)
# v21's premise ledger asked whether the derivation SPENT the stated quantities.
# Its cheapest form, per unit: does every number the unit states appear in the
# text of the samples that gave the consensus answer? It cannot see a condition
# without a number ("they forgot their wallets ... walk back"), which is where
# deletion can.
_NUM = re.compile(r'\d+(?:,\d{3})*(?:\.\d+)?')


def numbers(text: str) -> List[float]:
    return [float(n.replace(',', '')) for n in _NUM.findall(text)]


def spent(unit_text: str, raws: Sequence[str]) -> Optional[bool]:
    """True when every number of the unit occurs in at least one of `raws`;
    None for a unit without digits."""
    need = numbers(unit_text)
    if not need:
        return None
    have = {round(x, 9) for r in raws for x in numbers(r)}
    return all(round(x, 9) in have for x in need)


# ---------------------------------------------------------------- the detector
def same_answer(a: Optional[float], b: Optional[float]) -> bool:
    return a is not None and b is not None and abs(a - b) < INVARIANT_TOL


def flagged(us: Sequence[Dict], full: Optional[float], answers: Sequence[Optional[float]],
            kinds: Sequence[str] = PRIMARY_KINDS, numeric: bool = False) -> List[int]:
    """Indices of the informative units of `kinds` whose deletion left the answer
    unchanged (numeric=True: only units with a digit). A variant with no
    parsable answer is never invariant; a row with no full answer flags nothing."""
    out = []
    for j, (u, a) in enumerate(zip(us, answers)):
        if u['kind'] not in kinds or not u['informative']:
            continue
        if numeric and not re.search(r'\d', u['text']):
            continue
        if same_answer(full, a):
            out.append(j)
    return out


def eligible(us: Sequence[Dict], kinds: Sequence[str] = PRIMARY_KINDS) -> List[int]:
    """The units the flag can fire on."""
    return [j for j, u in enumerate(us) if u['kind'] in kinds and u['informative']]


# ---------------------------------------------------------------- statistics (exact, no randomness)
def fisher_greater(a: int, b: int, c: int, d: int) -> float:
    """One-sided Fisher exact p for [[a, b], [c, d]]: P(X >= a), X the first row's
    count in the first column, margins fixed."""
    n1, n2, k = a + b, c + d, a + c
    total = comb(n1 + n2, k)
    return sum(comb(n1, x) * comb(n2, k - x) for x in range(a, min(n1, k) + 1)) / total if total else 1.0


def chance_hit(n: int, changed: int, m: int) -> float:
    """P(at least one of m units drawn at random from n is among the `changed` ones)."""
    if m <= 0 or changed <= 0:
        return 0.0
    if n - changed < m:
        return 1.0
    return 1.0 - comb(n - changed, m) / comb(n, m)


def tail_at_least(ps: Sequence[float], k: int) -> float:
    """P(S >= k) for S a sum of independent Bernoulli(p_i) (Poisson-binomial, exact)."""
    dist = [1.0]
    for p in ps:
        nxt = [0.0] * (len(dist) + 1)
        for s, q in enumerate(dist):
            nxt[s] += q * (1 - p)
            nxt[s + 1] += q * p
        dist = nxt
    return sum(dist[k:]) if k > 0 else 1.0


def auroc(pos: Sequence[float], neg: Sequence[float]) -> Optional[float]:
    """P(a random positive scores above a random negative), ties count half."""
    if not pos or not neg:
        return None
    wins = sum((x > y) + 0.5 * (x == y) for x in pos for y in neg)
    return wins / (len(pos) * len(neg))
