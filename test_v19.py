"""[v19.0] Guards for the transplant pre-test. Offline, no models.

Run as `python test_v19.py`.
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
    print()
    print("PART 1 - the twin stays divisible, which is what v16.2 called granularity")
    # 25% of the twin, and 60% of that, must not produce fractions of people
    r = ST.scale_text("Of the 5611617 riders, 25% stay upright and 60% of those are women.")
    check(r.ok, "a twin is built")
    v = list(r.mapping.values())[0]
    check(v % r.smooth == 0, f"the twin value {v:g} is a multiple of {r.smooth}")
    check((v * 0.25) == int(v * 0.25), f"25% of {v:g} is whole")
    check((v * 0.25 * 0.6) == int(v * 0.25 * 0.6), f"60% of that is whole too")

    # halves and thirds
    r2 = ST.scale_text("A robe takes 2287720 bolts of blue fiber and half that much white.")
    v2 = list(r2.mapping.values())[0]
    check(r2.ok and (v2 / 2) == int(v2 / 2), f"half of {v2:g} is whole")

    # the snap never buys divisibility at more than the allowed distortion
    worst = 0.0
    with open(MANIFEST, encoding="utf-8") as fh:
        rows = json.load(fh)["rows"]
    for row in rows:
        s = ST.scale_text(row["text"])
        if not s.ok:
            continue
        for orig, tw in s.mapping.items():
            exact = orig / s.divisor
            worst = max(worst, abs(tw - exact) / exact)
    check(worst <= ST.MAX_SNAP_DISTORTION + 1e-9,
          f"no twin value is distorted by more than {100*ST.MAX_SNAP_DISTORTION:.0f}% "
          f"(worst {100*worst:.1f}%)")
    check(all(ST.scale_text(r["text"]).ok for r in rows),
          "all 40 pretest rows still get a twin after the snap")


def part2():
    print()
    print("PART 2 - the adapter hands v16.2's machinery a valid ShrinkResult")
    import magnitude_invariance as MI
    s = ST.scale_text("Samantha is now 3473626 and Raymond was born 6 years earlier.")
    sr = ST.as_shrink_result(s)
    check(isinstance(sr, MI.ShrinkResult), "an actual ShrinkResult comes back")
    check(sr.shrunk_text == s.twin_text, "the shrunk text is the twin text")
    twin_v = list(s.mapping.values())[0]
    check(sr.probe_to_original.get(twin_v) == 3473626.0,
          "the twin value maps back to the real number")

    # the rebind restores the real value
    rebound, ok, reason = MI.rebind_guarded({"age": twin_v}, sr)
    check(ok and rebound["age"] == 3473626.0, f"a given carrying the twin value is rebound ({reason})")

    # and the guard REFUSES when the twin value never reached the givens --
    # this is the load-bearing part: without it the chain would evaluate at the
    # twin's number and silently return a small answer for a large problem
    rebound2, ok2, reason2 = MI.rebind_guarded({"age": twin_v + 5}, sr)
    check(not ok2, f"a folded-away input makes the guard abstain ({reason2[:60]}...)")

    # small constants must NOT be touched
    rebound3, ok3, _ = MI.rebind_guarded({"age": twin_v, "years": 6}, sr)
    check(ok3 and rebound3["years"] == 6, "unscaled constants survive the rebind")


def part3():
    print()
    print("PART 3 - the pretest runs end to end offline")
    out = os.path.join(HERE, "_t_v19.json")
    r = subprocess.run([sys.executable, os.path.join(HERE, "pretest_v19.py"),
                        "--stub", "--limit", "5", "--out", out],
                       capture_output=True, text=True, cwd=HERE)
    check(r.returncode == 0, "pretest exits 0 in stub mode")
    if r.returncode != 0:
        print(r.stdout[-1500:], r.stderr[-1500:])
    else:
        data = json.load(open(out, encoding="utf-8"))
        rows = data["rows"]
        check(len(rows) == 5, "five rows recorded")
        check(all("W_correct" in x and "T_correct" in x and "O_correct" in x for x in rows),
              "every row carries all three arms")
        # the transplant path really ran: the twin's given came back as the
        # ORIGINAL number, through the real rebind and the real CAS
        ok_rebind = [x for x in rows if x.get("T_givens_rebound")]
        check(len(ok_rebind) == 5, "the rebind ran on every row")
        check(all(abs(float(list(x["T_givens_rebound"].values())[0]) - x["max_number"]) < 1e-6
                  for x in ok_rebind),
              "the rebound given is the ORIGINAL large number, not the twin's")
        check(all(x["T_answer"] is not None for x in ok_rebind),
              "the CAS evaluated the rebound blueprint")
        check("transplant" in r.stdout and "oracle" in r.stdout and "control" in r.stdout,
              "the summary names all three arms")
        check("no verdict" in r.stdout, "no verdict is printed at n=5")
    try:
        os.remove(out)
    except OSError:
        pass


def part4():
    print()
    print("PART 4 - abstention falls back to plain CoT, not to a wrong answer")
    src = open(os.path.join(HERE, "pretest_v19.py"), encoding="utf-8").read()
    check("T_correct=bool(r['cot_correct'])" in src,
          "an abstaining transplant scores as the recorded CoT result")
    check("T_covered" in src and "the rows it covered" in src,
          "coverage is reported separately from the deployable number")
    for token in ("T >= 36", "W >= 36", "O <= 33"):
        check(token in src, f"threshold {token!r} is written down before the run")


def main() -> int:
    part1(); part2(); part3(); part4()
    print()
    print("=" * 60)
    if FAILS:
        print(f"FAILURES ({len(FAILS)} of {N[0]}):")
        for f in FAILS:
            print("   " + f)
        return 1
    print(f"ALL {N[0]} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
