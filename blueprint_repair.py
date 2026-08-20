"""
Deterministic blueprint repair — mechanical defects, fixed without an LLM.

WHY THIS EXISTS
---------------
v13.0 and v14.0 both answered a failed SIV audit the same way: hand the error
report back to the Mathematician and ask for a fresh derivation. Measured
fix-rate: 6/39 (15.4%) then 9/51 (18%). Asking a 7B model to re-derive throws
away a derivation that is usually *conceptually fine* and gambles on a better
one.

Classifying every structural defect in the v14.0 run (25 rows, the first run
that recorded blueprint contents at all) shows why that gamble is a bad trade —
almost none of them are reasoning errors:

  ~9  unbalanced subscript:  givens['hotel_nights'] * givens['hotel_cost'
                             -> the ']' is simply missing
  ~10 near-miss name:        givens['peach_bought']   vs declared 'peaches_bought'
                             givens['monthly_hanger'] vs declared 'honthly_hanger'
                             total_cost_pick          vs computed 'total_cost_picking'
  ~3  computed var read as a given:
                             number_of_intervals = ...          (computed on line 1)
                             total = givens['number_of_intervals']  (line 2)
  ~2  numeric value stored as a string: {"number_of_houses": "3"}

Every one of those is a *transcription* defect with a single mechanically
correct answer. This module fixes them in microseconds, deterministically,
before any LLM repair is considered — so the expensive path is reserved for
defects that genuinely need re-derivation.

SAFETY
------
A wrong fuzzy match would convert a loudly-broken blueprint into a silently-wrong
one, which is strictly worse than leaving it broken. So name resolution requires
BOTH a high similarity score AND a unique winner; ties and weak matches are left
alone for the LLM path. Every repair is returned in a log so the CSV records what
was changed and nothing happens invisibly.
"""

import difflib
import json
import re
from typing import Any, Dict, List, Optional, Tuple

# A name must be at least this similar to a declared given/computed variable
# before we dare rewrite it. 0.82 accepts 'peach_bought' -> 'peaches_bought'
# and 'honthly_hanger_rental' -> 'monthly_hanger_rental' (single-character and
# short-suffix slips, which is what the model actually produces) while refusing
# genuinely different names.
_FUZZY_CUTOFF = 0.82

# The winner must beat the runner-up by this margin, otherwise the reference is
# ambiguous and we must not guess. Real case from the run: 'snails_aquariumt'
# sits between 'snails_aquarium1' and 'snails_aquarium2' -- picking either would
# be a coin flip dressed up as a repair.
_FUZZY_MARGIN = 0.05

_GIVENS_REF = re.compile(r"givens\[\s*(['\"])(.*?)\1\s*\]")


def _coerce_number(v: Any) -> Optional[float]:
    """A numeric value, or None if the value is not numerically meaningful."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip().replace(",", "").replace("$", "").replace("%", "")
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _balance_subscripts(eq: str) -> Tuple[str, int]:
    """Close `givens['name'` references that never got their ']'.

    Scans for the opening form and inserts the bracket right after the closing
    quote when it is missing. Counted and returned so the caller can log it.
    """
    out, i, fixed = [], 0, 0
    pat = re.compile(r"givens\[\s*(['\"])(.*?)\1")
    while True:
        m = pat.search(eq, i)
        if not m:
            out.append(eq[i:])
            break
        end = m.end()
        rest = eq[end:]
        stripped = rest.lstrip()
        if stripped.startswith("]"):
            out.append(eq[i:end])
            i = end
        else:
            out.append(eq[i:end] + "]")
            fixed += 1
            i = end
    return "".join(out), fixed


def _resolve_name(name: str, candidates: List[str]) -> Optional[str]:
    """Unique, high-confidence match for `name` among `candidates`, else None."""
    if not candidates or name in candidates:
        return None
    scored = sorted(
        ((difflib.SequenceMatcher(None, name, c).ratio(), c) for c in candidates),
        reverse=True,
    )
    best_score, best = scored[0]
    if best_score < _FUZZY_CUTOFF:
        return None
    if len(scored) > 1 and best_score - scored[1][0] < _FUZZY_MARGIN:
        return None  # ambiguous — refuse to guess
    return best


_FAMILY = re.compile(r"^(?P<stem>.*?)(?P<tail>[0-9A-Za-z])$")


def _resolve_by_sibling_exclusion(name, candidates, already_used):
    """[v14.8] Resolve a one-character-corrupted key inside an indexed family.

    The run produces keys where a single trailing character is wrong:
    'pages_chaptert' for 'pages_chapter2', 'snails_aquariumt' for
    'snails_aquarium2'. v14.2 deliberately refused these -- 'snails_aquariumt'
    is equidistant from 'snails_aquarium1' and 'snails_aquarium2', so fuzzy
    matching alone is a coin flip dressed up as a repair, and _FUZZY_MARGIN
    rejects it.

    Context breaks the tie. If the corrupted name shares a stem with several
    declared givens and all but ONE of those siblings is already referenced in
    the same equation, the remaining sibling is the only reading that does not
    duplicate a term. Real case: 'total_pages = givens[pages_chapter1] +
    givens[pages_chaptert] + givens[pages_chapter3]' over a family of
    {1,2,3} leaves exactly 'pages_chapter2'.

    Returns the resolved name, or None when the family is absent, the stem does
    not match, or more than one sibling is still free.
    """
    m = _FAMILY.match(name)
    if not m:
        return None
    stem = m.group("stem")
    if len(stem) < 3:
        return None
    family = [c for c in candidates
              if c != name and c.startswith(stem) and len(c) == len(name)]
    if len(family) < 2:
        return None
    free = [c for c in family if c not in already_used]
    return free[0] if len(free) == 1 else None


def _normalise_equations(equations):
    """[v14.8] Strip mechanical noise before any semantic repair is attempted.

    Three defects observed in mas_full_20260817 that make an otherwise sound
    chain unevaluable:
      * a whole statement block stuffed into one equation string, import and
        all: 'import math\nrounded_packages = math.ceil(packages_needed)'
      * a stray apostrophe welded to an identifier:
        'total_cost = cost_for_11_hours + additional_cost''
      * markdown fences and list bullets surviving JSON extraction
    """
    out, fixes = [], []
    for idx, raw in enumerate(equations):
        text = str(raw)
        stripped = re.sub(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$", "", text)
        if stripped != text:
            fixes.append(f"eq{idx}: removed markdown fence")
        for part in stripped.split(chr(10)):
            line = part.strip()
            if not line:
                continue
            if re.match(r"^(import|from)\s+[A-Za-z_]", line):
                fixes.append(f"eq{idx}: dropped {line!r} (evaluator injects math)")
                continue
            bullet = re.sub(r"^[-*]\s+", "", line)
            if bullet != line:
                fixes.append(f"eq{idx}: removed list bullet")
                line = bullet
            # The lookbehind is load-bearing: without it this eats the
            # CLOSING quote of every givens['key'] reference, because the
            # identifier there is also followed by a quote then ']'.
            dequoted = re.sub(
                r"(?<![A-Za-z_0-9'\"])([A-Za-z_][A-Za-z_0-9]*)'(?![A-Za-z_0-9])",
                r"\1", line)
            if dequoted != line:
                fixes.append(f"eq{idx}: removed stray apostrophe on identifier")
                line = dequoted
            out.append(line)
    if len(out) != len(equations):
        fixes.append(f"split {len(equations)} equation strings into {len(out)}")
    return out, fixes


# [v15.0] Numbers that may legitimately appear in an equation chain without
# being quoted in the problem text: unit conversions, halves, percents.
_STRUCTURAL_CONSTANTS = frozenset({
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 12.0, 100.0, 0.5,
    60.0, 24.0, 7.0, 365.0, 1000.0,
})

_TEXT_NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")


def _text_numbers(problem_text):
    """Every numeric literal in the problem statement, in order."""
    out = []
    for m in _TEXT_NUMBER.finditer(problem_text or ""):
        try:
            out.append(float(m.group(0).replace(",", "")))
        except ValueError:
            pass
    return out


def _digits(x):
    return str(int(x)) if float(x).is_integer() else str(x)


def _one_digit_apart(a, b):
    """True when the two numbers differ by a single digit substitution,
    insertion, or deletion -- the shape of a transcription slip."""
    a, b = _digits(a), _digits(b)
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(1 for i, j in zip(a, b) if i != j) == 1
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    return any(long_[:i] + long_[i + 1:] == short for i in range(len(long_)))


def _snap_givens_to_text(givens, problem_text):
    """[v15.0] Force every numeric given to be a number the problem ACTUALLY
    contains.

    The dominant defect in this pipeline is not reasoning, it is transcription
    of the operands. Measured on the 25-problem 5-route pretest
    (pretest_blueprint_consistency.json, 2026-08-19), 35% of extracted numeric
    givens do not appear anywhere in the problem text, and 68% of those sit one
    digit away from a number that does:

        "Katelyn saw 50 fairies ... 30 fairies flew away"  ->  55 and 31
        "Kim raises $320 more than Alexandra, who raises $430" -> 321 and 431
        "569 girls and 236 boys"                           ->  69 and 36

    The equations around these are frequently correct, so the whole derivation
    is lost to a misread digit. Snapping is deliberately conservative: a value
    is only rewritten when EXACTLY ONE text number is a single-digit slip away
    (or, failing that, exactly one text number shares its digit suffix, which
    catches a dropped leading digit). Ambiguity means no rewrite.

    Measured effect on those 125 blueprints: 37 correct -> 53 correct
    (+12.8pp), with ZERO previously-correct blueprints broken. On the
    production prompt alone, 28% -> 44%.
    """
    text_nums = _text_numbers(problem_text)
    if not text_nums:
        return givens, []
    present = set(text_nums)
    out, fixes = dict(givens), []
    for key, value in givens.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        v = float(value)
        if v in present or v in _STRUCTURAL_CONSTANTS:
            continue
        near = {t for t in text_nums if _one_digit_apart(v, t)}
        how = "one digit apart"
        if len(near) != 1:
            near = {t for t in text_nums
                    if _digits(t).endswith(_digits(v)) or _digits(v).endswith(_digits(t))}
            how = "shares digit suffix"
        if len(near) != 1:
            continue                      # ambiguous or unrelated: leave it alone
        target = near.pop()
        out[key] = target
        fixes.append(f"givens[{key!r}] {value} -> {target} (not in the problem "
                     f"text; {how} from a number that is)")
    return out, fixes


def repair_blueprint(blueprint: dict, problem_text: str = "") -> Tuple[dict, List[str]]:
    """Return (repaired_blueprint, fixes). The input is never mutated.

    `fixes` is empty when nothing mechanical was wrong, which is the caller's
    signal that the defect (if any) needs the LLM repair path instead.
    """
    givens = dict(blueprint.get("givens", {}) or {})
    equations = list(blueprint.get("equations", []) or [])
    if not equations:
        return blueprint, []

    fixes: List[str] = []

    # ── 0. Mechanical noise: statement blocks, stray quotes, fences [v14.8] ──
    equations, norm_fixes = _normalise_equations(equations)
    fixes.extend(norm_fixes)
    if not equations:
        return blueprint, []

    # ── 0.5 Operands that the problem text does not contain [v15.0] ──────
    if problem_text:
        givens, snap_fixes = _snap_givens_to_text(givens, problem_text)
        fixes.extend(snap_fixes)

    # ── 1. Numeric values stored as strings ──────────────────────────────────
    # SIV filters givens to int/float, so {"number_of_houses": "3"} makes every
    # equation touching it unauditable even though the value is perfectly usable.
    for k, v in list(givens.items()):
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            n = _coerce_number(v)
            if n is not None:
                givens[k] = n
                fixes.append(f"coerced givens[{k!r}] {v!r} -> {n}")

    numeric_keys = [k for k, v in givens.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)]

    # ── 2. Unbalanced subscripts ─────────────────────────────────────────────
    for idx, eq in enumerate(equations):
        new_eq, n = _balance_subscripts(str(eq))
        if n:
            equations[idx] = new_eq
            fixes.append(f"closed {n} unbalanced givens[...] in eq{idx}")

    # ── 3 & 4. Name resolution, walking the chain in order ───────────────────
    # `computed` grows as we go so a reference can only resolve against names
    # that already exist at that point -- the same rule the evaluator uses.
    computed: List[str] = []
    for idx, eq in enumerate(equations):
        eq = str(eq)
        if "=" not in eq or eq.strip().startswith("#"):
            continue
        lhs, rhs = eq.split("=", 1)

        # [v14.8] Which declared givens this equation already reads, so
        # sibling exclusion can tell a corrupted index from a fresh one.
        used_here = {k for k in _GIVENS_REF.findall(rhs)
                     for k in [k[1]] if k in numeric_keys}

        # 3. givens['key'] whose key is not a declared given.
        def _fix_ref(m):
            q, key = m.group(1), m.group(2)
            if key in numeric_keys:
                return m.group(0)
            # A computed variable read as if it were a given: drop the wrapper.
            if key in computed:
                fixes.append(f"eq{idx}: givens[{key!r}] -> {key} (computed above)")
                return key
            tgt = _resolve_name(key, numeric_keys)
            if tgt:
                fixes.append(f"eq{idx}: givens[{key!r}] -> givens[{tgt!r}]")
                return f"givens[{q}{tgt}{q}]"
            tgt = _resolve_name(key, computed)
            if tgt:
                fixes.append(f"eq{idx}: givens[{key!r}] -> {tgt} (computed above)")
                return tgt
            # [v14.8] Fuzzy matching abstained. If the key belongs to an indexed
            # family and every sibling but one is already referenced in THIS
            # equation, the remaining sibling is the only non-duplicating read.
            tgt = _resolve_by_sibling_exclusion(key, numeric_keys, used_here)
            if tgt:
                fixes.append(f"eq{idx}: givens[{key!r}] -> givens[{tgt!r}] "
                             f"(only unused sibling of its family)")
                return f"givens[{q}{tgt}{q}]"
            return m.group(0)

        rhs_new = _GIVENS_REF.sub(_fix_ref, rhs)

        # 4. Bare names that resolve to nothing: typo'd references to a given or
        #    to an earlier result. Protect string literals and call targets so a
        #    function name is never rewritten as a variable.
        masked = _GIVENS_REF.sub(lambda m: " " * len(m.group(0)), rhs_new)
        masked = re.sub(r"'[^']*'|\"[^\"]*\"", lambda m: " " * len(m.group(0)), masked)
        masked = re.sub(r"([A-Za-z_][A-Za-z_0-9]*)\s*\(",
                        lambda m: " " * len(m.group(0)), masked)
        known = set(numeric_keys) | set(computed) | _SAFE_NAMES
        edits = []
        for m in re.finditer(r"[A-Za-z_][A-Za-z_0-9]*", masked):
            name = m.group(0)
            if name in known:
                continue
            tgt = _resolve_name(name, computed) or _resolve_name(name, numeric_keys)
            if tgt:
                repl = tgt if tgt in computed else f"givens['{tgt}']"
                edits.append((m.start(), m.end(), repl))
                fixes.append(f"eq{idx}: undefined {name!r} -> {repl}")
        for start, end, repl in reversed(edits):
            rhs_new = rhs_new[:start] + repl + rhs_new[end:]

        equations[idx] = lhs + "=" + rhs_new
        computed.append(lhs.strip())

    if not fixes:
        return blueprint, []

    repaired = dict(blueprint)
    repaired["givens"] = givens
    repaired["equations"] = equations
    repaired["_deterministic_fixes"] = fixes
    return repaired, fixes


# Names an equation may use without declaring: what the SIV/SymPy evaluators
# inject, kept in sync with SymbolicInverseVerifier._ALLOWED_FREE_NAMES.
_SAFE_NAMES = frozenset({
    "abs", "max", "min", "ceil", "floor", "sqrt", "round", "int", "float",
    "sum", "len", "pow", "divmod", "log", "log10", "exp", "pi", "e", "E",
    "Abs", "Max", "Min", "N", "Rational", "math", "givens",
    "True", "False", "None", "and", "or", "not", "if", "else", "for", "in",
})
