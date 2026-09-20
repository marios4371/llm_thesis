"""[v18.0] Guards for scaled-twin prompting. Offline, no models.

Run as `python test_scaled_twin.py`.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import scaled_twin as ST

FAILS = []
N = [0]

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "pretest_data/clean_large_number_rows.json")


def check(cond, label):
    N[0] += 1
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        FAILS.append(label)


def part1():
    print("\nPART 1 - uniform scaling")
    t = ("Fred made 25956168 gallons of root beer. His children drank 5956168 "
         "of those gallons. Barbie spilled 7 gallons. 3 people came.")
    r = ST.scale_text(t)
    check(r.ok and r.n_scaled == 2, "both large numbers scaled, small ones untouched")
    check("7 gallons" in r.twin_text and "3 people" in r.twin_text, "7 and 3 survive")

    # the comma AFTER a number belongs to the sentence, not to the number
    rp = ST.scale_text("If Samantha is now 3473626, how many years ago, exactly?")
    check(rp.ok and ", how many years ago, exactly?" in rp.twin_text,
          "punctuation after a scaled number survives")
    # ...but a thousands separator INSIDE one is part of it
    rk = ST.scale_text("The fund holds 3,473,626 dollars and pays 40 people.")
    check(rk.ok and 3473626.0 in rk.mapping, "3,473,626 is read as one number")
    a, b = r.mapping[25956168.0], r.mapping[5956168.0]
    check(a > b, "order is preserved: the children still drink LESS than was made")
    check(abs(a / b - 25956168 / 5956168) < 0.15, "the ratio is approximately preserved")
    check(b >= 10, "the smaller large number keeps at least two digits")
    check(r.divisor == 1e4, f"divisor is a power of ten ({r.divisor:g})")
    check(b >= 10 * (7 + 3), "the twin stays clear of the small constants (7, 3)")

    # coreference: the same number twice maps to the same twin value
    t2 = "She has 3473626 cards. Later she gives 3473626 cards away and buys 100."
    r2 = ST.scale_text(t2)
    vals = [x for x in ST.NUMBER_RE.findall(r2.twin_text)]
    check(vals[0] == vals[1] and vals[2] == "100", "equal values map to equal twin values")

    # prefix safety: 5072217 inside 5072217640 must not be corrupted
    t3 = "He buys 5072217 pairs and spends $5072217640 a year."
    r3 = ST.scale_text(t3)
    check(r3.ok and r3.n_scaled == 2 and "5072217640" not in r3.twin_text
          and "5072217" not in r3.twin_text,
          "a number that is a prefix of another is replaced independently")

    # no large numbers -> refuse, text unchanged
    r4 = ST.scale_text("Janet has 16 eggs and sells 9 for $2 each.")
    check(not r4.ok and r4.twin_text == "Janet has 16 eggs and sells 9 for $2 each.",
          "a small-number problem gets no twin and is left alone")

    # two large numbers that would collide at the coarse scale separate at a finer one
    t5 = "Team A scored 1234567 and team B scored 4234589 points."
    r5 = ST.scale_text(t5)
    check(r5.ok and len(set(r5.mapping.values())) == 2,
          "large numbers that would collide are separated by keeping more digits")

    # ...but when separating them needs six digits the twin is still unreadable,
    # so there is no twin worth presenting
    t5b = "Team A scored 1234567 and team B scored 1234589 points."
    r5b = ST.scale_text(t5b)
    check(not r5b.ok and "under" in r5b.refused,
          "a twin that cannot get under the threshold is refused, not shipped")

    # percentages and decimals below the threshold stay literal
    t6 = "Of the 5611617 riders, 25% stayed upright and 0.5 of the rest fell."
    r6 = ST.scale_text(t6)
    check("25%" in r6.twin_text and "0.5" in r6.twin_text, "25% and 0.5 are untouched")


def _twin_answer(gold_code, mapping):
    """Run the gsm-hard gold program with the twin's numbers substituted, so the
    test can see what Problem A actually asks. Test-only: a deployed system has
    no gold program."""
    import re
    body = re.sub(r'""".*?"""', '"""doc"""', gold_code, flags=re.S)

    def repl(m):
        try:
            v = float(m.group(0).replace(",", ""))
        except ValueError:
            return m.group(0)
        if v in mapping:
            sv = mapping[v]
            return str(int(sv)) if float(sv).is_integer() else repr(sv)
        return m.group(0)

    src = ST.NUMBER_RE.sub(repl, body)
    ns = {}
    exec(src, ns)
    return ns["solution"]()


def part1b():
    print()
    print("PART 1b - the twin has to stay a possible question")
    # Scaling only the big numbers mixes two scales. These three rows are the
    # ones that broke: the shipped rule put Theo on $37 with $100 suits.
    with open(MANIFEST, encoding="utf-8") as fh:
        rows = {r["problem_id"]: r for r in json.load(fh)["rows"]}
    for pid in ("gsm-hard_229", "gsm-hard_1138", "gsm-hard_1280"):
        r = rows[pid]
        sc = ST.scale_text(r["text"])
        check(sc.ok, f"{pid}: a twin is still built")
        ans = _twin_answer(r["gold_code"], sc.mapping)
        check(ans > 0, f"{pid}: the twin has a positive answer ({ans}) -- was negative")
        check(max(sc.mapping.values()) < ST.DEFAULT_THRESHOLD,
              f"{pid}: the twin is under the collapse boundary")

    # the constraint itself, on a minimal case: $9,000,000 spent 40 at a time
    r = ST.scale_text("He has 9000000 dollars and buys 6 things at 40 dollars each.")
    v = list(r.mapping.values())[0]
    check(r.ok and v >= 10 * (6 + 40),
          f"the scaled amount ({v:g}) clears ten times the constants it is spent on")
    check(r.floor == 10 * 46, f"the floor is recorded ({r.floor:g})")

    # no small numbers at all -> nothing to stay clear of, scale all the way
    r2 = ST.scale_text("The city has 8123456 residents. How many is that?")
    check(r2.ok and max(r2.mapping.values()) < 100,
          "with no constants in the way the twin scales all the way down")


def part2():
    print("\nPART 2 - the prompt and reading the answers back")
    p = ST.twin_prompt("Problem with 56 riders.", "Problem with 5611617 riders.")
    check(p.index("Problem A") < p.index("Problem B"), "twin comes first")
    check("Answer B:" in p and "Answer A:" in p, "both answer lines are requested")

    a, b = ST.extract_answers("blah\nAnswer A: 14\nmore work\nAnswer B: 1,402,904\n")
    check(a == 14.0 and b == 1402904.0, "reads both answers, comma-aware")

    a, b = ST.extract_answers("**Answer A**: 14\n\n**Answer B**: 1402904")
    check(a == 14.0 and b == 1402904.0, "tolerates markdown bold around the labels")

    a, b = ST.extract_answers("Answer A: 14\nFor B: \\boxed{1402904}")
    check(b == 1402904.0, "falls back to boxed for B")

    a, b = ST.extract_answers("Answer A: 14\nSo the answer is 99.")
    check(a == 14.0 and b is None,
          "with no B marker the twin's answer is NOT mistaken for the real one")

    a, b = ST.extract_answers("Answer A: 14\nAnswer B: -7.5\nAnswer B: 8")
    check(b == 8.0, "the LAST Answer B wins")


def part3():
    print("\nPART 3 - the pretest runs end to end offline")
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "_stub_scaled_twin.json")
    r = subprocess.run([sys.executable, "pretest_scaled_twin.py", "--stub",
                        "--limit", "5", "--out", out],
                       capture_output=True, text=True, cwd=here)
    check(r.returncode == 0, "pretest exits 0 in stub mode" + ("" if r.returncode == 0 else "\n" + r.stderr[-800:]))
    if r.returncode == 0:
        data = json.load(open(out, encoding="utf-8"))
        check(data["n"] == 5 and all("S_correct" in x for x in data["rows"]),
              "every row carries the scaled-arm result")
        check(all(x.get("O_correct") for x in data["rows"]),
              "the stub oracle arm scores 5/5 (it answers with the gold)")
        check("arm S" in r.stdout and "arm O" in r.stdout and "control" in r.stdout,
              "the summary prints all three lines")
    try:
        os.remove(out)
    except OSError:
        pass

    # the manifest itself
    man = json.load(open(os.path.join(here, "pretest_data/clean_large_number_rows.json"),
                         encoding="utf-8"))
    check(len(man["rows"]) == 40, f"manifest has 40 rows ({len(man['rows'])})")
    check(all(r["original_text"] and r["max_number"] >= 1e5 for r in man["rows"]),
          "every row has an original text and a large number")
    check(sum(r["cot_correct"] for r in man["rows"]) == 32, "recorded control is 32/40")
    twins = [ST.scale_text(r["text"]) for r in man["rows"]]
    check(sum(t.ok for t in twins) == 40, "a twin can be built for all 40 rows")


def main() -> int:
    part1(); part1b(); part2(); part3()
    print("\n" + "=" * 60)
    if FAILS:
        print(f"FAILURES ({len(FAILS)} of {N[0]}):")
        for f in FAILS:
            print("   " + f)
        return 1
    print(f"ALL {N[0]} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
