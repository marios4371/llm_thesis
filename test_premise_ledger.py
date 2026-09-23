"""
[v21.0] Offline checks for premise_ledger.py. No model, no network.

Run:  python test_premise_ledger.py
"""
import sys
from collections import Counter

import premise_ledger as L

RABBITS = ("Daphne went to her allotment to pick some raspberries and found two "
           "times as many wasps as beetles in the patch. Daphne then found 64 white "
           "animals, half of which were rabbits. In addition, she saw 28 "
           "caterpillars, and 12 beetles. What percentage of animals in the patch "
           "were rabbits?")
FARMER = ("A farmer buys six 4-pound boxes of hay, twelve 5-pound bags of beets, "
          "thirty one 40-pound packs of corn and eighteen 17-pound sacks of apples. "
          "Finally, he buys 43 pounds of groceries, and 27 pounds of tools, and 100 "
          "pounds of wood. A farm truck can carry 120 pounds at a time. If the "
          "farmer has three trucks, how many trips does the farmer need?")
POPCORN = ("80 pop in the first 16 seconds of cooking, then 2 times that amount in "
           "the next 16 seconds. In the final 16 seconds, the popping slows down to "
           "half the rate. How many popped?")

FAILS = []


def check(name, cond, detail=''):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not cond else ''))
    if not cond:
        FAILS.append(name)


def surfaces(ps):
    return [p.surface for p in ps]


print("PART 1  premises")
p = L.extract_premises(RABBITS)
check("rabbits: 'two', 64, 'half', 28, 12 all extracted",
      surfaces(p) == ['two', '64', 'half', '28', '12'], surfaces(p))
p = L.extract_premises(FARMER)
check("farmer: spelled 'thirty one' is 31", any(q.value == 31 for q in p))
check("farmer: 'three' trucks is a premise", 'three' in surfaces(p), surfaces(p))
p = L.extract_premises(POPCORN)
check("popcorn: the three '16 seconds' merge into one premise",
      surfaces(p).count('16') == 1 and next(q for q in p if q.surface == '16').sentences == (0, 1),
      [(q.surface, q.sentences) for q in p])
p = L.extract_premises("A shirt costs $20 and is 25% off.")
q = next(x for x in p if x.value == 25)
check("25% is a percent premise spendable as 0.75", q.kind == 'percent' and 0.75 in q.forms)
p = L.extract_premises("She drank two-thirds of the 12 bottles.")
q = next(x for x in p if x.kind == 'fraction')
check("'two-thirds' spendable as 2 or 3 (written '* 2 / 3')", 2.0 in q.forms and 3.0 in q.forms)
p = L.extract_premises("On the third day he read one of the books, 9 pages.")
check("ordinal 'third' and pronoun 'one' are not quantities", surfaces(p) == ['9'], surfaces(p))


print("\nPART 2  operands")
cot = ("Oscar has 96 puppies.\n"
       "Spotted: \\[ \\frac{64}{2} = 32 \\]\n"
       "Total: 64 + 28 + 12 = 104\n"
       "Again, \\[ \\frac{64}{2} = 32 \\]")
ops = L.cot_operands(cot).counter()
check("prose restatement is not an operand (96)", 96.0 not in ops)
check("\\frac{64}{2} gives 64, 2 and 32", ops[64.0] >= 1 and ops[2.0] == 1 and ops[32.0] >= 1, dict(ops))
check("an identical restated block counts once", ops[2.0] == 1, dict(ops))
code = "x = 3  # 99 in a comment\nprint('7 apples', x * 5)\n"
ops = L.code_operands(code).counter()
check("code: literals only, comments and strings ignored",
      set(ops) == {3.0, 5.0}, dict(ops))


print("\nPART 3  audit")
prem = L.extract_premises(RABBITS)
wrong = ("Rabbits: \\[ \\frac{64}{2} = 32 \\]\n"
         "Total: \\[ 64 + 28 + 12 = 104 \\]\n"
         "\\[ \\frac{32}{104} \\times 100 = 30.77 \\]")
right = ("Wasps: \\[ 2 \\times 12 = 24 \\]\n"
         "Rabbits: \\[ \\frac{64}{2} = 32 \\]\n"
         "Total: \\[ 64 + 28 + 12 + 24 = 128 \\]\n"
         "\\[ \\frac{32}{128} \\times 100 = 25 \\]")
aw, ar = L.audit(prem, L.cot_operands(wrong)), L.audit(prem, L.cot_operands(right))
check("the wasp-dropping CoT leaves exactly one premise unused", aw.n_unconsumed == 1)
check("...and it is 'two' (sentence 0), not the 'half' spent next to 64",
      [q.short() for q in aw.unconsumed] == ["'two'@s0"], [q.short() for q in aw.unconsumed])
check("the complete CoT spends everything", ar.complete, [q.short() for q in ar.unconsumed])
folded = ("Wasps are 24.\n\\[ 64 + 28 + 12 + 24 = 128 \\]\n\\[ \\frac{64}{2} = 32 \\]\n"
          "\\[ \\frac{32}{128} \\times 100 = 25 \\]")
af = L.audit(prem, L.cot_operands(folded))
check("folding 2 x 12 into '24' is credited by one-hop", af.complete,
      [q.short() for q in af.unconsumed])

prem = L.extract_premises("He buys eighteen 17-pound sacks and twelve 5-pound bags. "
                          "Each trip carries 120 pounds. How many trips?")
ops = Counter({18.0: 1, 12.0: 1, 5.0: 1, 120.0: 1, 6.0: 1})
a = L.audit(prem, ops)
check("one-hop never steals a directly-stated operand (17 unused, not 120)",
      [q.surface for q in a.unconsumed] == ['17'], [q.short() for q in a.unconsumed])


print("\nPART 4  the vote")
prem = L.extract_premises(RABBITS)
aud = [L.audit(prem, L.cot_operands(x)) for x in (wrong, wrong, right)]
ans, info = L.premise_vote([30.77, 30.77, 25.0], aud)
check("a complete minority outvotes an incomplete majority", ans == 25.0 and info['moved'], info)
ans, info = L.premise_vote([25.0, 25.0, 30.77], [aud[2], aud[2], aud[0]])
check("an incomplete minority cannot move a complete majority", ans == 25.0 and not info['moved'])
ans, info = L.premise_vote([30.77, 31.0, 30.77], [aud[0], aud[0], aud[0]])
check("equal coverage everywhere is plain majority", ans == 30.77 and not info['moved'])
ans, info = L.premise_vote([10.0, 20.0, 30.0], [aud[0], aud[2], aud[2]])
check("a tie among the eligible goes to the first seen of them", ans == 20.0)
ans, info = L.premise_vote([None, None], [aud[0], aud[0]])
check("no answers -> None", ans is None and not info['moved'])
shared = L.shared_omissions(prem, [aud[0], aud[0]])
check("shared_omissions names the clause every sample skipped",
      [q.short() for q in shared] == ["'two'@s0"], [q.short() for q in shared])
check("shared_omissions is empty once any sample spent it",
      L.shared_omissions(prem, [aud[0], aud[2]]) == [])


print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("all checks passed")
