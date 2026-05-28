"""
Comparative Evaluation Metrics for MAS-SHT
===========================================
Companion to Mas_solver.py v10.2 + baselines.py.

Three public functions:

    compute_all_metrics(results_dict, reference_system) -> pd.DataFrame
        Per-system summary: accuracy, deltas, efficiency, SIV/SHT rates.

    run_mcnemar_tests(results_dict, reference_system) -> pd.DataFrame
        Paired McNemar's test (MAS-SHT vs each baseline) on the
        intersection of problem_ids. Auto-selects exact vs Yates.

    plot_comparison(metrics_df, output_path) -> Path
        3-panel figure: accuracy, efficiency Pareto, Δ with significance.

Apples-to-apples enforcement:
    All comparisons intersect problem_id sets across systems before
    computing. Rows that exist for one system but not all systems are
    DROPPED from the comparison and the surviving N is reported as
    `n_paired`. This keeps the McNemar contingency table well-defined
    and stops asymmetric exhaustion (e.g., Groq daily limit hit on
    system X) from biasing the result.

Schema expected for each system's DataFrame:
    Required columns: problem_id, correct (bool/int), answer (float|str),
                      time_s (float), num_llm_calls (int),
                      tokens_estimated (int)
    Optional MAS columns: siv_invertible, siv_verified,
                          siv_execution_audit_passed,
                          sht_triggered (bool)
    Optional error column: error_type (str — "" on success)

The runner in MAS_SHT_Experiments.ipynb produces this schema directly.
"""

from __future__ import annotations

import os
import math
import logging
from typing import Dict, Optional, Tuple, List
import numpy as np
import pandas as pd

# statsmodels is already a project dep (used for SIV stats elsewhere).
try:
    from statsmodels.stats.contingency_tables import mcnemar
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

logger = logging.getLogger("MAS_Pipeline.metrics")


# =====================================================================
# Internal helpers
# =====================================================================

def _safe_mean(s: pd.Series) -> float:
    """Mean over non-null numeric entries; 0 on empty."""
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.mean()) if len(s) else 0.0


# =====================================================================
# Confidence intervals for proportions (accuracy)
# =====================================================================
#
# Two complementary methods:
#   wilson_ci      — closed-form, used for the headline accuracy CIs in the
#                    summary table. Asymptotically valid down to n≈30 and
#                    well-behaved at p near 0 or 1 (unlike the normal-approx
#                    Wald interval). Matches what NLP venues report as
#                    "95% CI" for accuracy.
#   bootstrap_ci   — non-parametric resampling, used when we need CIs on
#                    derived statistics (delta, error reduction) where the
#                    closed form does not generalise. Also exposed for
#                    accuracy so callers can sanity-check Wilson.
#
# Both default to alpha=0.05 (95% CI).

def wilson_ci(correct: int, n: int, alpha: float = 0.05) -> Tuple[float, float, float]:
    """Wilson score interval for a proportion.

    Returns (point_estimate, lower, upper). Falls back to (0, 0, 0) when n=0.
    """
    if n <= 0:
        return 0.0, 0.0, 0.0
    p = correct / n
    # z for two-sided 1-alpha. For alpha=0.05 → z≈1.959964.
    # Avoid scipy dep for this single value; use erf-based inverse normal.
    z = _z_for_alpha(alpha)
    denom = 1.0 + (z * z) / n
    centre = (p + (z * z) / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def _z_for_alpha(alpha: float) -> float:
    """Two-sided z critical value via the inverse-normal approximation.
    Accurate to ~6 decimals for the alphas we care about (0.01, 0.05, 0.10).
    Avoids a scipy dependency for this hot path."""
    # Beasley-Springer-Moro is overkill here; hardcode the common ones
    # and fall back to a Newton step on math.erf for the rest.
    table = {0.10: 1.6448536269514722,
             0.05: 1.959963984540054,
             0.01: 2.5758293035489004}
    if alpha in table:
        return table[alpha]
    # Newton from a normal-approx start
    p = 1.0 - alpha / 2.0
    x = 0.0
    for _ in range(50):
        # CDF Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))
        cdf = 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
        pdf = math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
        if pdf < 1e-12:
            break
        x += (p - cdf) / pdf
    return x


def bootstrap_ci(values: np.ndarray, alpha: float = 0.05,
                 n_boot: int = 2000,
                 seed: int = 0,
                 stat=np.mean) -> Tuple[float, float, float]:
    """Non-parametric percentile bootstrap CI for an arbitrary statistic.

    Args:
        values: 1-D array of per-problem observations (0/1 for accuracy,
                real-valued for time/cost).
        alpha:  two-sided coverage level (0.05 → 95% CI).
        n_boot: number of resamples (2000 is enough for the second-decimal
                stability we need; 10000 if you want sub-percent stability).
        seed:   PRNG seed for reproducibility.
        stat:   any function array → scalar. Default is mean (accuracy).
    Returns:
        (point_estimate, lower, upper). All NaN if values is empty.
    """
    values = np.asarray(values).astype(float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(stat(values))
    rng = np.random.default_rng(seed)
    n = values.size
    samples = np.empty(n_boot)
    # Vectorised resample: draws an (n_boot, n) index matrix in one shot.
    idx = rng.integers(0, n, size=(n_boot, n))
    for i in range(n_boot):
        samples[i] = stat(values[idx[i]])
    lo = float(np.quantile(samples, alpha / 2.0))
    hi = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return point, lo, hi


def paired_delta_bootstrap_ci(values_a: np.ndarray, values_b: np.ndarray,
                              alpha: float = 0.05, n_boot: int = 2000,
                              seed: int = 0) -> Tuple[float, float, float]:
    """Bootstrap CI for the paired difference mean(a) - mean(b).

    Uses paired resampling (same indices drawn for both vectors) to preserve
    the within-problem correlation — this is the right thing for "system A vs
    system B on the same problem set". Reporting an unpaired CI here would
    overestimate the variance.

    Args:
        values_a, values_b: equal-length per-problem observations.
        alpha: two-sided coverage (0.05 → 95% CI on the delta).
        n_boot, seed: as in bootstrap_ci.
    Returns:
        (delta_point, delta_lower, delta_upper). NaN if vectors empty or
        of unequal length.
    """
    a = np.asarray(values_a).astype(float)
    b = np.asarray(values_b).astype(float)
    if a.size == 0 or a.size != b.size:
        return float("nan"), float("nan"), float("nan")
    point = float(a.mean() - b.mean())
    rng = np.random.default_rng(seed)
    n = a.size
    deltas = np.empty(n_boot)
    idx = rng.integers(0, n, size=(n_boot, n))
    for i in range(n_boot):
        sel = idx[i]
        deltas[i] = a[sel].mean() - b[sel].mean()
    lo = float(np.quantile(deltas, alpha / 2.0))
    hi = float(np.quantile(deltas, 1.0 - alpha / 2.0))
    return point, lo, hi


def _intersect_problem_ids(results_dict: Dict[str, pd.DataFrame]) -> List[str]:
    """Compute the intersection of problem_id across all systems. This is
    the apples-to-apples problem set used for every cross-system metric."""
    if not results_dict:
        return []
    sets = [set(df["problem_id"].astype(str)) for df in results_dict.values()
            if "problem_id" in df.columns and len(df)]
    if not sets:
        return []
    common = sets[0]
    for s in sets[1:]:
        common &= s
    # Sort for determinism (downstream test reproducibility).
    return sorted(common)


def _restrict_to_ids(df: pd.DataFrame, ids: List[str]) -> pd.DataFrame:
    """Restrict and dedupe to the given problem_ids; keep the latest row
    if duplicates exist (e.g., re-runs from checkpointing)."""
    if "problem_id" not in df.columns:
        return df.iloc[0:0]
    sub = df[df["problem_id"].astype(str).isin(ids)].copy()
    if "timestamp" in sub.columns:
        sub = sub.sort_values("timestamp").drop_duplicates("problem_id", keep="last")
    else:
        sub = sub.drop_duplicates("problem_id", keep="last")
    sub = sub.set_index(sub["problem_id"].astype(str)).loc[ids]
    return sub.reset_index(drop=True)


# =====================================================================
# 1) Per-system metrics
# =====================================================================

def compute_all_metrics(results_dict: Dict[str, pd.DataFrame],
                        reference_system: str = "mas_sht_full",
                        ci_alpha: float = 0.05,
                        bootstrap_n: int = 2000,
                        bootstrap_seed: int = 0) -> pd.DataFrame:
    """
    Compute per-system metrics over the apples-to-apples problem intersection.

    Args:
        results_dict: {system_name: df} where each df has at minimum
            problem_id, correct, time_s, num_llm_calls, tokens_estimated.
            MAS systems additionally carry siv_*/sht_* columns.
        reference_system: the system whose accuracy is treated as the
            "candidate" — Δ_accuracy and error_reduction are computed
            relative to each OTHER system as the baseline (so for the
            reference row those columns are 0/NaN).
        ci_alpha: two-sided coverage for the CIs (0.05 → 95% CI).
        bootstrap_n: bootstrap resamples for the paired-delta CI.
        bootstrap_seed: PRNG seed for the bootstrap; fixed for reproducibility.

    Returns:
        DataFrame with one row per system, columns:
            system, n_total, n_paired, accuracy,
            accuracy_ci_lo, accuracy_ci_hi,   # [v10.4] Wilson 95% CI
            delta_vs_ref, delta_ci_lo, delta_ci_hi,  # [v10.4] paired bootstrap CI
            error_reduction_vs_ref,
            avg_llm_calls, avg_tokens, avg_time_s,
            accuracy_per_call,
            siv_trigger_rate, siv_skip_rate,
            sht_trigger_rate, acc_when_sht_triggered, acc_when_sht_skipped
    """
    cols = [
        "system", "n_total", "n_paired", "accuracy",
        "accuracy_ci_lo", "accuracy_ci_hi",
        "delta_vs_ref", "delta_ci_lo", "delta_ci_hi",
        "error_reduction_vs_ref",
        "avg_llm_calls", "avg_tokens", "avg_time_s",
        "accuracy_per_call",
        "siv_trigger_rate", "siv_skip_rate",
        "sht_trigger_rate", "acc_when_sht_triggered", "acc_when_sht_skipped",
    ]
    if not results_dict:
        return pd.DataFrame(columns=cols)

    common = _intersect_problem_ids(results_dict)
    if not common:
        logger.warning("compute_all_metrics: no shared problem_ids across systems.")

    rows = []
    ref_acc: Optional[float] = None
    ref_correct_vec: Optional[np.ndarray] = None

    # First pass: compute reference accuracy AND the per-problem correctness
    # vector (needed for paired-delta bootstrap below).
    if reference_system in results_dict and common:
        ref_df = _restrict_to_ids(results_dict[reference_system], common)
        if len(ref_df):
            ref_acc = _safe_mean(ref_df["correct"])
            ref_correct_vec = (
                pd.to_numeric(ref_df["correct"], errors="coerce")
                  .fillna(0).astype(int).to_numpy()
            )

    for system, df in results_dict.items():
        n_total = len(df)
        sub = _restrict_to_ids(df, common) if common else df
        n_paired = len(sub)
        acc = _safe_mean(sub["correct"]) if n_paired else 0.0

        # [v10.4] Wilson 95% CI for headline accuracy.
        n_correct = int(pd.to_numeric(sub["correct"], errors="coerce")
                            .fillna(0).astype(int).sum()) if n_paired else 0
        _, acc_lo, acc_hi = wilson_ci(n_correct, n_paired, alpha=ci_alpha)

        # Paired delta vs reference (point + bootstrap CI).
        # We compare per-problem correctness vectors; this captures the
        # within-problem correlation that an unpaired bound would miss.
        delta = 0.0
        delta_lo, delta_hi = float("nan"), float("nan")
        if ref_acc is not None and system != reference_system and ref_correct_vec is not None:
            sys_corr_vec = (
                pd.to_numeric(sub["correct"], errors="coerce")
                  .fillna(0).astype(int).to_numpy()
            )
            if sys_corr_vec.shape == ref_correct_vec.shape:
                # delta = system − reference (positive = system beats reference)
                delta, delta_lo, delta_hi = paired_delta_bootstrap_ci(
                    sys_corr_vec, ref_correct_vec,
                    alpha=ci_alpha, n_boot=bootstrap_n, seed=bootstrap_seed,
                )

        # Error reduction: (acc - ref) / (1 - ref).
        if ref_acc is not None and system != reference_system and ref_acc < 1.0:
            err_red = (acc - ref_acc) / (1.0 - ref_acc)
        else:
            err_red = float("nan")

        avg_calls = _safe_mean(sub["num_llm_calls"]) if "num_llm_calls" in sub.columns else 0.0
        avg_toks = _safe_mean(sub["tokens_estimated"]) if "tokens_estimated" in sub.columns else 0.0
        avg_time = _safe_mean(sub["time_s"]) if "time_s" in sub.columns else 0.0
        apc = (acc / avg_calls) if avg_calls > 0 else float("nan")

        # SIV-specific: only meaningful for MAS variants.
        siv_trig = float("nan")
        siv_skip = float("nan")
        if "siv_invertible" in sub.columns and len(sub):
            siv_trig = _safe_mean(sub["siv_invertible"].fillna(False).astype(bool).astype(int))
        if "siv_verified" in sub.columns and "sht_triggered" in sub.columns and len(sub):
            siv_verified = sub["siv_verified"].fillna(False).astype(bool)
            sht_skipped  = ~sub["sht_triggered"].fillna(False).astype(bool)
            denom = max(int(siv_verified.sum()), 1)
            siv_skip = float((siv_verified & sht_skipped).sum()) / denom

        # SHT-specific
        sht_rate = float("nan")
        acc_when_sht = float("nan")
        acc_when_no_sht = float("nan")
        if "sht_triggered" in sub.columns and len(sub):
            sht_mask = sub["sht_triggered"].fillna(False).astype(bool)
            sht_rate = float(sht_mask.mean())
            if sht_mask.any():
                acc_when_sht = _safe_mean(sub.loc[sht_mask, "correct"])
            if (~sht_mask).any():
                acc_when_no_sht = _safe_mean(sub.loc[~sht_mask, "correct"])

        rows.append({
            "system": system,
            "n_total": n_total,
            "n_paired": n_paired,
            "accuracy": acc,
            "accuracy_ci_lo": acc_lo,
            "accuracy_ci_hi": acc_hi,
            "delta_vs_ref": delta,
            "delta_ci_lo": delta_lo,
            "delta_ci_hi": delta_hi,
            "error_reduction_vs_ref": err_red,
            "avg_llm_calls": avg_calls,
            "avg_tokens": avg_toks,
            "avg_time_s": avg_time,
            "accuracy_per_call": apc,
            "siv_trigger_rate": siv_trig,
            "siv_skip_rate": siv_skip,
            "sht_trigger_rate": sht_rate,
            "acc_when_sht_triggered": acc_when_sht,
            "acc_when_sht_skipped": acc_when_no_sht,
        })

    out = pd.DataFrame(rows)
    # Put reference at the top, then sort the rest by accuracy desc.
    if reference_system in out["system"].values:
        ref_row = out[out["system"] == reference_system]
        rest = out[out["system"] != reference_system].sort_values("accuracy", ascending=False)
        out = pd.concat([ref_row, rest], ignore_index=True)
    return out


# =====================================================================
# 2) McNemar's test, paired
# =====================================================================

def run_mcnemar_tests(results_dict: Dict[str, pd.DataFrame],
                      reference_system: str = "mas_sht_full",
                      alpha: float = 0.05) -> pd.DataFrame:
    """
    Paired McNemar's test: reference_system vs each other system over the
    intersection of problem_ids.

    Auto-selects test variant based on the discordant-pair count (b+c):
        b+c <  25  → exact binomial (no continuity correction)
        b+c >= 25  → asymptotic chi-square with Yates' correction
    Rationale: the asymptotic test is unreliable for small discordant
    counts; the exact test is computationally cheap at any scale we use.

    Returns columns:
        comparison, n_paired, a, b, c, d,
        statistic, p_value, significant_at_alpha, test_used
    where (a, b, c, d) is the contingency table:
        a = both correct
        b = ref correct, other wrong          (ref's "wins")
        c = ref wrong,   other correct        (ref's "losses")
        d = both wrong
    """
    cols = ["comparison", "n_paired", "a", "b", "c", "d",
            "statistic", "p_value", "significant_at_alpha", "test_used"]
    if reference_system not in results_dict:
        logger.warning(f"run_mcnemar_tests: reference '{reference_system}' missing.")
        return pd.DataFrame(columns=cols)
    if not STATSMODELS_AVAILABLE:
        logger.warning("statsmodels not installed; McNemar skipped.")
        return pd.DataFrame(columns=cols)

    common = _intersect_problem_ids(results_dict)
    if not common:
        return pd.DataFrame(columns=cols)

    ref_df = _restrict_to_ids(results_dict[reference_system], common)
    ref_corr = ref_df["correct"].astype(bool).to_numpy()

    rows = []
    for system, df in results_dict.items():
        if system == reference_system:
            continue
        other_df = _restrict_to_ids(df, common)
        if len(other_df) != len(ref_df):
            logger.warning(
                f"McNemar {system}: row count mismatch after restrict — skipping."
            )
            continue
        other_corr = other_df["correct"].astype(bool).to_numpy()
        a = int(((ref_corr) & (other_corr)).sum())   # both correct
        b = int(((ref_corr) & (~other_corr)).sum())  # ref correct, other wrong
        c = int(((~ref_corr) & (other_corr)).sum())  # ref wrong, other correct
        d = int(((~ref_corr) & (~other_corr)).sum())  # both wrong

        discordant = b + c
        # statsmodels mcnemar takes a 2x2 array.
        table = np.array([[a, b], [c, d]])
        if discordant == 0:
            # Degenerate: every problem agrees. p=1, test is uninformative.
            stat, p, used = 0.0, 1.0, "degenerate_no_discordant"
        elif discordant < 25:
            res = mcnemar(table, exact=True, correction=False)
            stat, p, used = float(res.statistic), float(res.pvalue), "exact_binomial"
        else:
            res = mcnemar(table, exact=False, correction=True)
            stat, p, used = float(res.statistic), float(res.pvalue), "asymptotic_yates"

        rows.append({
            "comparison": f"{reference_system}_vs_{system}",
            "n_paired": len(ref_df),
            "a": a, "b": b, "c": c, "d": d,
            "statistic": stat,
            "p_value": p,
            "significant_at_alpha": bool(p < alpha),
            "test_used": used,
        })

    out = pd.DataFrame(rows, columns=cols)
    return out


# =====================================================================
# 3) Plotting
# =====================================================================

def plot_comparison(metrics_df: pd.DataFrame,
                    mcnemar_df: Optional[pd.DataFrame] = None,
                    output_path: str = "comparison.png",
                    reference_system: str = "mas_sht_full",
                    title_suffix: str = "") -> str:
    """
    Three-panel figure:
        (1) Accuracy bar chart, sorted, reference highlighted.
        (2) Efficiency scatter — accuracy vs avg_llm_calls per system.
            Pareto frontier dashed (top-left = best).
        (3) Δ-accuracy vs reference, with significance stars from McNemar
            if mcnemar_df is supplied.

    Saves to output_path and returns the path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if metrics_df is None or len(metrics_df) == 0:
        logger.warning("plot_comparison: empty metrics_df, skipping.")
        return ""

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"MAS-SHT Comparative Evaluation{(' — ' + title_suffix) if title_suffix else ''}",
                 fontsize=14)

    # -------- Panel 1: Accuracy --------
    ax = axes[0]
    df1 = metrics_df.sort_values("accuracy", ascending=True)
    colors = ["#d62728" if s == reference_system else "#1f77b4" for s in df1["system"]]
    ax.barh(df1["system"], df1["accuracy"], color=colors, edgecolor="black")
    ax.set_xlabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_title("Accuracy by System")
    for i, (s, v) in enumerate(zip(df1["system"], df1["accuracy"])):
        ax.text(v + 0.01, i, f"{v:.1%}", va="center", fontsize=9)

    # -------- Panel 2: Efficiency --------
    ax = axes[1]
    for _, r in metrics_df.iterrows():
        is_ref = r["system"] == reference_system
        ax.scatter(r["avg_llm_calls"], r["accuracy"],
                   s=140 if is_ref else 90,
                   c="#d62728" if is_ref else "#1f77b4",
                   edgecolors="black",
                   marker="*" if is_ref else "o",
                   label=r["system"])
        ax.annotate(r["system"], (r["avg_llm_calls"], r["accuracy"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)
    # Pareto frontier (top-left = high acc, low cost). Take points that
    # are not dominated.
    pts = sorted(zip(metrics_df["avg_llm_calls"], metrics_df["accuracy"]))
    pareto = []
    best = -1.0
    for x, y in pts:
        if y > best:
            pareto.append((x, y))
            best = y
    if len(pareto) >= 2:
        px, py = zip(*pareto)
        ax.plot(px, py, "--", color="grey", alpha=0.6, label="Pareto frontier")
    ax.set_xlabel("Avg LLM calls per problem")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Compute Cost")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # -------- Panel 3: Δ + significance --------
    ax = axes[2]
    df3 = metrics_df[metrics_df["system"] != reference_system].copy()
    df3 = df3.sort_values("delta_vs_ref")
    if mcnemar_df is not None and len(mcnemar_df):
        # Map "ref_vs_other" comparison to other system name and p-value.
        pmap = {}
        for _, row in mcnemar_df.iterrows():
            comp = row["comparison"]
            if comp.startswith(f"{reference_system}_vs_"):
                other = comp[len(f"{reference_system}_vs_"):]
                pmap[other] = row["p_value"]
        df3["p_value"] = df3["system"].map(pmap)
    else:
        df3["p_value"] = float("nan")

    deltas = df3["delta_vs_ref"].to_numpy()
    bar_colors = ["#2ca02c" if d > 0 else "#d62728" for d in deltas]
    ax.barh(df3["system"], deltas, color=bar_colors, edgecolor="black")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"Δ Accuracy vs {reference_system} (negative = ref wins)")
    ax.set_title(f"{reference_system} − Baseline (signed)")
    # Significance stars
    for i, (sys_name, d, p) in enumerate(zip(df3["system"], deltas, df3["p_value"])):
        stars = ""
        if not (p is None or (isinstance(p, float) and math.isnan(p))):
            if p < 0.001:
                stars = "***"
            elif p < 0.01:
                stars = "**"
            elif p < 0.05:
                stars = "*"
        if stars:
            ax.text(d + (0.005 if d >= 0 else -0.005), i, stars,
                    va="center",
                    ha="left" if d >= 0 else "right",
                    fontsize=11, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"plot_comparison: saved {output_path}")
    return output_path


# =====================================================================
# Convenience: write the canonical artifact bundle
# =====================================================================

def write_artifact_bundle(results_dict: Dict[str, pd.DataFrame],
                          out_dir: str = ".",
                          reference_system: str = "mas_sht_full") -> Dict[str, str]:
    """One-shot helper that writes:
        comparison_table.csv, comparison_table.md, mcnemar_results.csv,
        comparison.png
    Returns a {artifact_name: path} dict."""
    os.makedirs(out_dir, exist_ok=True)
    metrics = compute_all_metrics(results_dict, reference_system)
    mcnemar = run_mcnemar_tests(results_dict, reference_system)

    paths = {}
    p = os.path.join(out_dir, "comparison_table.csv")
    metrics.to_csv(p, index=False)
    paths["comparison_csv"] = p

    p = os.path.join(out_dir, "comparison_table.md")
    try:
        with open(p, "w", encoding="utf-8") as f:
            f.write(metrics.to_markdown(index=False, floatfmt=".4f"))
    except Exception as e:
        logger.warning(f"to_markdown failed (need 'tabulate'): {e}")
    paths["comparison_md"] = p

    p = os.path.join(out_dir, "mcnemar_results.csv")
    mcnemar.to_csv(p, index=False)
    paths["mcnemar_csv"] = p

    p = os.path.join(out_dir, "comparison.png")
    plot_comparison(metrics, mcnemar, output_path=p, reference_system=reference_system)
    paths["comparison_png"] = p

    return paths
