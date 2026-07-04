"""Aggregate per-system CSVs downloaded from the split Kaggle commits into the
thesis tables — no GPU / model / network needed. Run in your local venv.

Workflow (split-per-commit plan):
  1. Kaggle commit with RUN_PHASE='baselines'  -> download its results CSV(s)
  2. Kaggle commit with RUN_PHASE='mas_full'   -> download its results CSV
  3. Kaggle commit with RUN_PHASE='mas_no_siv' -> download its results CSV
  4. Put ALL downloaded *.csv into one folder, then:

       python aggregate_local.py <folder>

Outputs <folder>/artifacts/: comparison_table.csv, mcnemar_results.csv,
siv_evaluation_report.md (+ the per-table SIV CSVs).
"""
import sys, os, glob
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # so evaluation_metrics / siv_evaluation import

folder = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = os.path.join(folder, "artifacts")
os.makedirs(OUT, exist_ok=True)

csvs = [c for c in sorted(glob.glob(os.path.join(folder, "*.csv")))
        if os.path.basename(os.path.dirname(c)) != "artifacts"]
print(f"Found {len(csvs)} CSV(s):")
for c in csvs:
    print("  ", os.path.basename(c))
if not csvs:
    sys.exit("No CSVs found in folder.")

merged = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
merged["timestamp"] = pd.to_datetime(merged.get("timestamp"), errors="coerce")
merged = (merged.sort_values("timestamp")
                .drop_duplicates(["problem_id", "system"], keep="last")
                .reset_index(drop=True))

EXPECTED = {"qwen_math_7b_local", "Qwen/Qwen2.5-Math-7B-Instruct"}
if "preset" in merged.columns:
    n0 = len(merged)
    merged = merged[merged["preset"].isin(EXPECTED)].reset_index(drop=True)
    if n0 - len(merged):
        print(f"[filter] dropped {n0 - len(merged)} row(s) from other presets")

results_dict = {s: g.reset_index(drop=True) for s, g in merged.groupby("system")}

summ = (merged.groupby("system")
        .agg(n=("problem_id", "nunique"), accuracy=("correct", "mean"),
             avg_time_s=("time_s", "mean"), avg_llm_calls=("num_llm_calls", "mean"))
        .sort_values("accuracy", ascending=False).round(4))
print("\n=== Summary ===")
print(summ.to_string())

# ── Stats (Wilson CI, paired bootstrap, McNemar) ──
try:
    from evaluation_metrics import compute_all_metrics, run_mcnemar_tests
    _mas = [s for s in results_dict if s.startswith("mas_sht_")]
    REF = "mas_sht_math7b" if "mas_sht_math7b" in results_dict else (_mas[0] if _mas else None)
    if REF:
        compute_all_metrics(results_dict, reference_system=REF).to_csv(
            os.path.join(OUT, "comparison_table.csv"), index=False)
        run_mcnemar_tests(results_dict, reference_system=REF).to_csv(
            os.path.join(OUT, "mcnemar_results.csv"), index=False)
        print(f"\nWrote comparison_table.csv + mcnemar_results.csv -> {OUT}")
    else:
        print("\nNo mas_sht_* system yet — stats skipped (run the MAS phases).")
except Exception as e:
    print("Stats step skipped:", e)

# ── SIV tables ──
try:
    import siv_evaluation as se
    _mas = [s for s in results_dict if s.startswith("mas_sht_")]
    if _mas:
        FULL = "mas_sht_math7b" if "mas_sht_math7b" in results_dict else _mas[0]
        NO_SIV = next((s for s in results_dict if "no_siv" in s), None)
        report = se.siv_report(results_dict[FULL], results_dict[NO_SIV] if NO_SIV else None)
        with open(os.path.join(OUT, "siv_evaluation_report.md"), "w", encoding="utf-8") as f:
            f.write(report)
        se.siv_coverage(results_dict[FULL]).to_csv(os.path.join(OUT, "siv_coverage.csv"), index=False)
        se.siv_detector_metrics(results_dict[FULL]).to_csv(os.path.join(OUT, "siv_detector.csv"), index=False)
        se.error_layer_decomposition(results_dict[FULL]).to_csv(os.path.join(OUT, "siv_error_layers.csv"), index=False)
        se.siv_skip_efficacy(results_dict[FULL]).to_csv(os.path.join(OUT, "siv_skip_efficacy.csv"), index=False)
        if NO_SIV:
            se.siv_ablation(results_dict[FULL], results_dict[NO_SIV]).to_csv(
                os.path.join(OUT, "siv_ablation.csv"), index=False)
        print(f"Wrote SIV tables + siv_evaluation_report.md -> {OUT}\n")
        print(report)
    else:
        print("No MAS run yet — SIV tables skipped.")
except Exception as e:
    print("SIV step skipped:", e)
