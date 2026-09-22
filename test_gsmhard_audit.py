"""[v20.0] Guards for the GSM-Hard construction audit. Offline, no datasets.

Every fixture below is a real GSM-Hard row, trimmed. If the audit ever stops
flagging these, the reported defect rate is no longer the one the paper cites.

Run as `python test_gsmhard_audit.py`.
"""
from __future__ import annotations

import sys

import gsmhard_audit as A

FAILS = []
N = [0]


def check(cond, label):
    N[0] += 1
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        FAILS.append(label)


# --- real rows -------------------------------------------------------------

# gsm-hard_933: "3 weeks" and "3 friends" in the original; the text took the
# substitution in both places, the gold program in only one.
R933_TEXT = ("Ryan's allowance is $6 each week he completes his chores. He did "
             "his chores for 8404276 weeks. Then he bought ice cream cones for "
             "himself and 8404276 friends at $1.25 each. Now they all want to "
             "go to the movies and tickets cost $6.50 each. How many movie "
             "tickets can Ryan buy?")
R933_CODE = '''
def solution():
    """Ryan's allowance is $6 each week he completes his chores. He did his chores for 3 weeks. Then he bought ice cream cones for himself and 3 friends at $1.25 each. Now they all want to go to the movies and tickets cost $6.50 each. How many movie tickets can Ryan buy?"""
    allowance_per_week = 6
    weeks_chores_done = 8404276
    allowance_total = allowance_per_week * weeks_chores_done
    ice_cream_cost = 1.25
    ice_cream_friends = 3
    result = allowance_total / 6.5
    return result
'''

# gsm-hard_775: "4 meatballs" and "Mark ate 4 subs" both became 1952796 in the
# text; the gold kept sidney_subs_eaten = 4. Eating 1952796 of 7 sandwiches is
# also not a thing that can happen.
R775_TEXT = ("One meatball sub sandwich contains 1952796 meatballs. Sidney "
             "ordered 3 less than ten meatball sub sandwiches. Then Mark ate "
             "1952796 of Sidney's meatball sub sandwiches. So Sidney ordered "
             "another three sub sandwiches. How many meatballs remained?")
R775_CODE = '''
def solution():
    """One meatball sub sandwich contains 4 meatballs. Sidney ordered 3 less than ten meatball sub sandwiches. Then Mark ate 4 of Sidney's meatball sub sandwiches. So Sidney ordered another three sub sandwiches. How many meatballs remained?"""
    meatballs_per_sub = 1952796
    sidney_subs_initial = 10 - 3
    sidney_subs_eaten = 4
    result = (sidney_subs_initial - sidney_subs_eaten + 3) * meatballs_per_sub
    return result
'''

# a correctly built row: the substitution landed in exactly one slot in both
CLEAN_TEXT = ("Raymond and Samantha are cousins. Raymond was born 6 years "
              "before Samantha. Raymond had a son at the age of 23. If "
              "Samantha is now 3473626, how many years ago was Raymond's son "
              "born?")
CLEAN_CODE = '''
def solution():
    """Raymond and Samantha are cousins. Raymond was born 6 years before Samantha. Raymond had a son at the age of 23. If Samantha is now 31, how many years ago was Raymond's son born?"""
    raymond_age_when_samantha_born = 6
    samantha_current_age = 3473626
    raymond_age_when_son_born = 23
    result = samantha_current_age - (raymond_age_when_son_born - raymond_age_when_samantha_born)
    return result
'''


def part1():
    print("\nPART 1 - the parts the audit is built from")
    check(A.original_question(CLEAN_CODE).startswith('Raymond and Samantha'),
          "the original GSM8K question is recovered from the docstring")
    check('31' in A.original_question(CLEAN_CODE),
          "...and it still carries the SMALL original value")
    check(A.perturbed_value(CLEAN_TEXT, CLEAN_CODE) == 3473626.0,
          "the substituted value is identified by diffing text against docstring")
    lits = A.code_literals(CLEAN_CODE)
    check(3473626.0 in lits and 23.0 in lits,
          "code literals are read from assignment right-hand sides")
    check(A.original_question('def f(): pass') == '',
          "a program with no docstring yields no original")
    check(A.perturbed_value(CLEAN_TEXT, 'def f(): pass') is None,
          "...and is therefore not auditable")

    # spelled quantities have to count, or gsm-hard_266's collision is invisible
    nums = A.numbers_in_text('he buys two boxes and 5 bags')
    check(2.0 in nums and 5.0 in nums, "'two' counts as an occurrence of 2")


def part2():
    print("\nPART 2 - the real broken rows are flagged")
    a = A.audit_row(R933_TEXT, R933_CODE)
    check(a['auditable'], "gsm-hard_933 is auditable")
    check(a['defect'] == 'collision', "gsm-hard_933 is a COLLISION")
    check(a['n_text'] == 2 and a['n_code'] == 1,
          f"...8404276 stands in 2 text slots and 1 code slot "
          f"(got {a['n_text']}/{a['n_code']})")

    b = A.audit_row(R775_TEXT, R775_CODE)
    check(b['defect'] == 'collision', "gsm-hard_775 is a COLLISION")
    check(b['n_text'] == 2 and b['n_code'] == 1,
          f"...2 text slots, 1 code slot (got {b['n_text']}/{b['n_code']})")


def part3():
    print("\nPART 3 - a correctly built row is NOT flagged")
    c = A.audit_row(CLEAN_TEXT, CLEAN_CODE)
    check(c['auditable'], "the clean row is auditable")
    check(c['defect'] is None, "the clean row carries no defect")
    check(c['n_text'] == 1 and c['n_code'] == 1,
          "one slot in the text, one in the gold -- which is the intended build")


def part4():
    print("\nPART 4 - the never-perturbed channel")
    code = CLEAN_CODE.replace('samantha_current_age = 3473626',
                              'samantha_current_age = 31')
    d = A.audit_row(CLEAN_TEXT, code)
    check(d['defect'] == 'never_perturbed',
          "a gold left at its GSM8K original is flagged separately")
    check(d['n_code'] == 0, "...because the large value is nowhere in the program")


def part5():
    print("\nPART 5 - the audit does not invent defects out of small variation")
    text = "Tom has 4 apples and 7 pears. How many pieces of fruit?"
    code = ('def solution():\n    """Tom has 4 apples and 7 pears. How many '
            'pieces of fruit?"""\n    result = 4 + 7\n    return result\n')
    e = A.audit_row(text, code)
    check(not e['auditable'],
          "an unperturbed problem is reported as not auditable, not as a defect")
    check(A.PERTURBATION_FLOOR >= 1000,
          "the floor keeps ordinary two-digit differences out of the count")


if __name__ == '__main__':
    for p in (part1, part2, part3, part4, part5):
        p()
    print(f"\n{N[0] - len(FAILS)}/{N[0]} checks passed")
    for f in FAILS:
        print("  FAILED: " + f)
    sys.exit(1 if FAILS else 0)
