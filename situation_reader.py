"""
[v24.0] The Reader agent: sentence-anchored situation QAs (ASQ) for a
separate Solver agent. Not a verifier: nothing is scored, nothing is
selected, and every row gets the same treatment.

THE PROBLEM IT IS BUILT FOR (measured, see v24_diagnosis.py)
------------------------------------------------------------
On GSM-Symbolic P2 the Solver's remaining errors are misreadings of the
situation, not arithmetic (the v21 role errors). They are systematic:
- 68% of the row-level accuracy variance lies between templates;
- the 10 worst of 49 templates hold 54% of all wrong samples;
- on 18 of B5's 22 errors no sample is right.
Resampling re-draws the same reading, so the gain left is in the reading
itself.

THE METHOD IT BUILDS ON, AND THE LIMITATION IT TAKES UP
------------------------------------------------------
MathWorld (Opedal et al., ACL Findings 2023) gives math story problems a
world model: containers plus TRANSFER / RATE / COMPARISON / PARTWHOLE
relations. From GOLD world models it generated question-answer pairs about
the situation. Inserting two such pairs right after the sentence they
describe raised GPT-3 from 70.8% to 78.6% (1-shot). The same pairs appended
at the end gave 71.8%. The gold annotation was the catch: the automatic
route, Codex parsing into world models, solved about a third of the easiest
dataset. The paper's Limitations call this "an obvious limitation" and leave
stronger parsers to future work.
Plan-and-Solve (Wang et al., ACL 2023) names the same gap from the other
side. Its Limitations: planning fixes calculation and missing-step errors,
"but the semantic misunderstanding errors still remain".

WHAT v24 DOES
-------------
It skips the formal parser. The QA pairs are the representation, as in
QA-based meaning representations, which people and LLMs produce far more
reliably than logical forms.
  Reader  Qwen2.5-7B-Instruct, greedy: the general-language model in the
          mixed preset (the thesis's role-by-competence split, v12.3). It
          sees the problem as numbered sentences. For each sentence it writes
          1-3 QA pairs making the situation explicit (whose each number is,
          every part of a split group including "the rest", the reference of
          each comparison, rates, order of changes), then one "Asked:" line.
          It never answers the final question.
  Solver  Qwen2.5-Math-7B-Instruct, the unchanged CoT prompt. The problem slot
          holds the problem with each sentence followed by the Reader's notes
          on it (ASQ).
Controls built from the SAME reading:
  END   the same notes appended after the unchanged problem (MathWorld's "all
        at once"): tests whether the position matters.
  SELF  one call: the Solver itself is told to read sentence by sentence
        first (the single-agent PS+/DUP route): tests whether a separate
        Reader matters.

WHAT IS AND IS NOT NEW (checked 2026-09-27)
-------------------------------------------
Not new: the relation types (MathWorld; schema-based instruction), situation
QAs placed at the sentence (MathWorld, gold only), understanding before
solving (PS+, DUP, one model, no sentence anchoring), a problem-level schema
label (SBI-RAG, one schema per problem, judged by an LLM rather than by
accuracy), decomposer + solver pairs (Socratic CoT, trained, sub-questions
are solution steps).
New here:
1. The sentence-anchored situation QAs come from a Reader agent, with no
   gold world model and no parser. This makes MathWorld's gold-only effect
   something a system can run.
2. Reading and solving are split across a general and a math-specialized
   model of the same size.
3. It is tested where semantic misunderstanding dominates (P2), with a 7B
   solver, at equal solver samples. It carries MathWorld's placement control
   (END) and a single-agent control (SELF).
Closest negative evidence: revealing a problem incrementally with a JSON
state collapses small models ("Beyond Pattern Recognition", 2025: Llama-3B
0.64 -> 0.05). v24 hides nothing. The Reader sees the whole problem, and so
does the Solver.

Everything in this file is deterministic and model-free, so it is tested
offline (test_v24.py). pretest_v24.py runs it.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

import premise_ledger as L
import pretest_v21 as V21

READER_MAX_TOKENS = 900
MAX_QA_PER_SENTENCE = 4
MAX_NOTE_CHARS = 320
MAX_ASKED_CHARS = 240

READER_SYSTEM = """You are the Reader in a two-agent team that solves math word problems. You never solve the problem; a separate Solver does that after you. Your job is to make the situation in every sentence explicit, so that the Solver cannot misread it.

You will see the problem split into numbered sentences S1, S2, ... For each sentence, write one to three short question-answer pairs that say what that sentence tells us:
- what each number counts or measures, whose it is, and its unit;
- when a group is split into parts, every part, including "the rest" (for example, "a class of 30, a third of whom wear glasses" means 10 students wear glasses and the other 20 do not);
- when a quantity is given relative to another ("twice as many as", "5 fewer than", "half of", "20% more than"), which quantity is the reference;
- rates ("per", "each", "every") and what they apply to;
- when something changes, the value before and after, and the order in which things happen.
You may do one simple calculation when a sentence directly implies a number. Never answer the problem's final question.
If a sentence gives no quantity or relation, write "S<n>: -".
End with one line "Asked: ..." that says exactly which quantity the final question asks for, and in which unit, without giving a number.

Write only these lines:
S1: Q: <question> A: <answer>
S2: -
Asked: <the quantity asked for>"""

# Few-shot demonstrations, adapted from GSM8K TRAIN problems. GSM-Symbolic
# templates are built from GSM8K TEST problems, so nothing here is a P2
# template. The demos show the four relation kinds the rules name: a split
# into parts, a comparison with its reference, a rate with its scope, and a
# change over time.
READER_DEMOS: Tuple[Tuple[str, str], ...] = (
    ("Mark has a garden with flowers. He planted plants of three different colors in it. "
     "Ten of them are yellow, and there are 80% more of those in purple. There are only 25% "
     "as many green flowers as there are yellow and purple flowers. How many flowers does "
     "Mark have in his garden?",
     "S1: -\n"
     "S2: Q: How are the flowers in the garden split? A: By color into three groups: every "
     "flower is yellow, purple or green.\n"
     "S3: Q: How many flowers are yellow? A: 10.\n"
     "S3: Q: How many flowers are purple, relative to what? A: 80% more than the yellow ones "
     "(the yellow flowers are the reference): 10 + 0.8 x 10 = 18.\n"
     "S4: Q: How many flowers are green, relative to what? A: 25% of the yellow and purple "
     "flowers together (the reference is 10 + 18 = 28): 0.25 x 28 = 7.\n"
     "S5: -\n"
     "Asked: the total number of flowers in the garden, counting all three colors."),
    ("Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, "
     "she read twice as many pages as yesterday. If she wants to read half of the remaining "
     "pages tomorrow, how many pages should she read?",
     "S1: Q: How many pages does the book have? A: 120.\n"
     "S2: Q: How many pages did she read yesterday? A: 12.\n"
     "S2: Q: How many pages did she read today, relative to what? A: Twice as many as "
     "yesterday (yesterday is the reference): 2 x 12 = 24.\n"
     "S3: Q: What are \"the remaining pages\"? A: The pages of the 120-page book that she has "
     "not read yet, after yesterday and today.\n"
     "S3: Q: How much of the remaining pages will she read tomorrow? A: Half of them.\n"
     "Asked: the number of pages she should read tomorrow."),
    ("Tina makes $18.00 an hour. If she works more than 8 hours per shift, she is eligible "
     "for overtime, which is paid by your hourly wage + 1/2 your hourly wage. If she works "
     "10 hours every day for 5 days, how much money does she make?",
     "S1: Q: What is Tina's normal pay rate? A: $18.00 for each hour worked.\n"
     "S2: Q: Which hours are overtime? A: Only the hours beyond the first 8 in a shift; the "
     "first 8 hours of each shift are paid at the normal rate.\n"
     "S2: Q: What is the overtime pay rate? A: The hourly wage plus half of it: "
     "18 + 9 = $27 per overtime hour.\n"
     "S3: Q: What are her shifts? A: 5 shifts (one per day), each 10 hours long, so each "
     "shift has 8 normal hours and 2 overtime hours.\n"
     "Asked: the total amount of money she makes over the 5 days, in dollars."),
)

ASQ_INTRO = "(Each line in parentheses is a careful reader's note on the sentence just above it.)"
END_HEADER = "Notes from a careful reader:"
SELF_PROMPT = (
    "Solve this math problem. First go through the problem sentence by sentence and, for "
    "each sentence, say what it tells us: what each number counts and whose it is, every "
    "part of any group that is split (including the rest), what each comparison is relative "
    "to, and any rates. Then solve it step by step. After your reasoning, state the final "
    "numeric answer on a line starting with 'Answer:'.\n\n"
    "Problem: {problem}\n\nLet's think step by step."
)


# ---------------------------------------------------------------------------
# the Reader's input
# ---------------------------------------------------------------------------

def sentences(text: str) -> List[str]:
    """The ledger's splitter (renderings.py), so a sentence index here is a
    sentence index in every earlier version."""
    return L.split_sentences(text)


def numbered(sents: Sequence[str]) -> str:
    return "Problem sentences:\n" + "\n".join(f"S{i}: {s}" for i, s in enumerate(sents, 1))


def reader_messages(text: str) -> List[Dict[str, str]]:
    msgs = [{'role': 'system', 'content': READER_SYSTEM}]
    for problem, reading in READER_DEMOS:
        msgs.append({'role': 'user', 'content': numbered(sentences(problem))})
        msgs.append({'role': 'assistant', 'content': reading})
    msgs.append({'role': 'user', 'content': numbered(sentences(text))})
    return msgs


# ---------------------------------------------------------------------------
# the Reader's output
# ---------------------------------------------------------------------------

_LINE = re.compile(r'^\s*(?:[-*•]\s*)?\**\s*S\s*(\d+)\s*\**\s*[:.)\-]\s*\**\s*(.*)$', re.I)
_ASKED = re.compile(r'^\s*(?:[-*•]\s*)?\**\s*asked\s*\**\s*[:\-]\s*\**\s*(.+)$', re.I)
_PAIR = re.compile(r'Q\s*:\s*(.*?)\s*A\s*:\s*(.*?)(?=\s*\bQ\s*:|$)', re.I | re.S)
_NONE = {'', '-', '--', '—', 'none', 'n/a', 'nothing', '(none)'}


def _clean(s: str, limit: int) -> str:
    s = re.sub(r'\s+', ' ', str(s)).strip()
    s = re.sub(r'^\*\*|\*\*$', '', s).strip()        # markdown bold around a whole field
    if len(s) > limit:
        s = s[:limit - 3].rstrip() + '...'
    return s


def parse_reading(raw: str, n_sentences: int) -> Dict:
    """The Reader's text -> {'notes': {i: [(q, a), ...]}, 'asked', 'ok', ...}.

    Sanitation, and nothing else. It keeps only the pairs anchored to a real
    sentence, at most MAX_QA_PER_SENTENCE per sentence. It drops an 'Asked'
    line that states a result ('='). It never judges whether a note is right:
    that would make the Reader's output a verifier's input."""
    notes: Dict[int, List[Tuple[str, str]]] = {}
    asked = ''
    dropped = 0
    for line in str(raw or '').splitlines():
        m = _ASKED.match(line)
        if m:
            cand = _clean(m.group(1), MAX_ASKED_CHARS)
            if '=' in cand:
                dropped += 1
            elif cand and not asked:
                asked = cand
            continue
        m = _LINE.match(line)
        if not m:
            continue
        i, rest = int(m.group(1)), m.group(2).strip()
        if not 1 <= i <= n_sentences:
            dropped += 1
            continue
        if rest.lower() in _NONE:
            continue
        pairs = [(_clean(q, MAX_NOTE_CHARS), _clean(a, MAX_NOTE_CHARS))
                 for q, a in _PAIR.findall(rest)]
        pairs = [(q, a) for q, a in pairs if a]
        if not pairs:
            pairs = [('', _clean(rest, MAX_NOTE_CHARS))]
        have = notes.setdefault(i, [])
        for p in pairs:
            if len(have) < MAX_QA_PER_SENTENCE:
                have.append(p)
            else:
                dropped += 1
    n_notes = sum(len(v) for v in notes.values())
    return {'notes': {str(k): [list(p) for p in v] for k, v in sorted(notes.items())},
            'asked': asked, 'n_notes': n_notes, 'dropped': dropped,
            'ok': bool(n_notes or asked)}


def _notes_of(reading: Dict, i: int) -> List[Tuple[str, str]]:
    return [tuple(p) for p in (reading.get('notes') or {}).get(str(i), [])]


def note_line(q: str, a: str) -> str:
    return f"(Q: {q} A: {a})" if q else f"({a})"


def asked_line(reading: Dict) -> str:
    return f"(Asked: {reading['asked']})" if reading.get('asked') else ''


def note_lines(text: str, reading: Dict) -> List[str]:
    """Every note, in sentence order, then the Asked line: the content both
    placements carry."""
    out = [note_line(q, a) for i in range(1, len(sentences(text)) + 1)
           for q, a in _notes_of(reading, i)]
    if asked_line(reading):
        out.append(asked_line(reading))
    return out


# ---------------------------------------------------------------------------
# the Solver's input, per arm
# ---------------------------------------------------------------------------

def interleaved(text: str, reading: Dict) -> str:
    """The problem with each sentence followed by the Reader's notes on it."""
    sents = sentences(text)
    out = [ASQ_INTRO]
    for i, s in enumerate(sents, 1):
        out.append(s)
        out.extend(note_line(q, a) for q, a in _notes_of(reading, i))
    if asked_line(reading):
        out.append(asked_line(reading))
    return '\n'.join(out)


def appended(text: str, reading: Dict) -> str:
    """The unchanged problem, then the same notes all at once."""
    return text + '\n\n' + END_HEADER + '\n' + '\n'.join(note_lines(text, reading))


def cot_prompt(text: str) -> str:
    """Arm C: exactly the prompt of every earlier version."""
    return V21.COT_PROMPT.format(problem=text)


def asq_prompt(text: str, reading: Dict) -> str:
    """The CoT prompt, with the annotated problem in the problem slot. A
    reading with nothing usable leaves the prompt byte-identical to C."""
    if not reading or not reading.get('ok'):
        return cot_prompt(text)
    return V21.COT_PROMPT.format(problem=interleaved(text, reading))


def end_prompt(text: str, reading: Dict) -> str:
    if not reading or not reading.get('ok'):
        return cot_prompt(text)
    return V21.COT_PROMPT.format(problem=appended(text, reading))


def self_prompt(text: str) -> str:
    return SELF_PROMPT.format(problem=text)


# ---------------------------------------------------------------------------
# telemetry (reported, never used by the method)
# ---------------------------------------------------------------------------

_NUM = re.compile(r"(?<![A-Za-z_])-?\d+(?:,\d{3})*(?:\.\d+)?")


def numbers(text: str) -> List[float]:
    out = []
    for t in _NUM.findall(str(text or '')):
        try:
            out.append(float(t.replace(',', '')))
        except ValueError:
            pass
    return out


def states_gold(reading: Dict, gold: Optional[float]) -> bool:
    """Does any note contain the gold value? If this is common, the Reader is
    solving rather than reading. Post hoc only: the method never sees gold."""
    if gold is None or not reading:
        return False
    vals = [v for ps in (reading.get('notes') or {}).values() for _, a in ps for v in numbers(a)]
    return any(V21.correct(v, gold) for v in vals)
