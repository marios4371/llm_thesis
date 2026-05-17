import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('final_results_v73_homogeneous_google_n50.csv')

print("=" * 60)
print("OVERALL SUMMARY")
print("=" * 60)
n = len(df)
bl = df.baseline_correct.sum()
ms = df.mas_correct.sum()
print(f"Total problems: {n}")
print(f"Baseline correct: {bl}/{n} ({bl/n*100:.1f}%)")
print(f"MAS correct:      {ms}/{n} ({ms/n*100:.1f}%)")
print(f"Delta:            {ms - bl} ({(ms-bl)/n*100:.1f}pp)")

print()
print("=" * 60)
print("REGRESSIONS (baseline=True, mas=False)")
print("=" * 60)
reg = df[(df.baseline_correct == True) & (df.mas_correct == False)]
print(f"Count: {len(reg)}")
if len(reg) > 0:
    for _, r in reg.iterrows():
        print(f"\n  ID: {r['id']}")
        print(f"  Dataset: {r['dataset']}")
        print(f"  Baseline answer: {r['baseline_ans']}")
        print(f"  MAS answer: {r['mas_ans']}")
        print(f"  Expected: {str(r['expected_snippet'])[:80]}")
        print(f"  SIV verified: {r.get('siv_verified', 'N/A')}")
        print(f"  SIV confidence: {r.get('siv_confidence', 'N/A')}")
        print(f"  SIV execution audit: {r.get('siv_execution_audit_passed', 'N/A')}")
        print(f"  SIV blueprint answer: {r.get('siv_blueprint_answer', 'N/A')}")
        print(f"  SIV rel error: {r.get('siv_execution_rel_error', 'N/A')}")
        print(f"  Solver agent: {r.get('solver_agent', 'N/A')}")
        print(f"  SHT triggered: {r.get('sht_triggered', 'N/A')}")
        print(f"  SHT triage: {r.get('sht_triage_result', 'N/A')}")
        print(f"  Verification passed: {r.get('verification_passed', 'N/A')}")
        print(f"  Verification confidence: {r.get('verification_confidence', 'N/A')}")

print()
print("=" * 60)
print("GAINS (baseline=False, mas=True)")
print("=" * 60)
gains = df[(df.baseline_correct == False) & (df.mas_correct == True)]
print(f"Count: {len(gains)}")
if len(gains) > 0:
    for _, r in gains.iterrows():
        print(f"  {r['id']}: baseline={r['baseline_ans']} -> mas={r['mas_ans']}")

print()
print("=" * 60)
print("BY DATASET")
print("=" * 60)
for ds in sorted(df.dataset.unique()):
    sub = df[df.dataset == ds]
    bl_d = sub.baseline_correct.sum()
    ms_d = sub.mas_correct.sum()
    nd = len(sub)
    delta = ms_d - bl_d
    print(f"  {ds:40s}: baseline={bl_d}/{nd} ({bl_d/nd*100:.0f}%), mas={ms_d}/{nd} ({ms_d/nd*100:.0f}%), delta={delta:+d}")

print()
print("=" * 60)
print("SIV STATISTICS")
print("=" * 60)
print(f"SIV verified True:  {(df.siv_verified == True).sum()}")
print(f"SIV verified False: {(df.siv_verified == False).sum()}")
print(f"SIV N/A:            {df.siv_verified.isna().sum()}")
print(f"SIV exec audit passed True:  {(df.siv_execution_audit_passed == True).sum()}")
print(f"SIV exec audit passed False: {(df.siv_execution_audit_passed == False).sum()}")

print()
print("SHT triggered:  ", (df.sht_triggered == True).sum())
print("SHT not triggered:", (df.sht_triggered == False).sum())

# Confident skip analysis
cs = df[df.sht_triage_result == 'confident_skip']
print(f"\nConfident skip (SIV gates SHT away): {len(cs)}")
print(f"  Of these, baseline correct: {cs.baseline_correct.sum()}")
print(f"  Of these, MAS correct:      {cs.mas_correct.sum()}")
print(f"  Regressions in confident_skip: {((cs.baseline_correct == True) & (cs.mas_correct == False)).sum()}")
