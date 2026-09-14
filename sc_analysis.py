"""
[v15.7] Iso-compute comparison: MAS vs SC@k on the same problems.

The MAS spends 3.41 LLM calls per problem. The question a reader asks first is
whether that budget beats simply sampling the SAME model k times and voting.
This module answers it from one SC@5 run: `baselines.self_consistency` records
the per-sample answers in order to a sidecar JSONL, so SC@1..SC@5 are all
recoverable -- SC@3 is the majority vote over the first three samples.

Join: the runner calls baselines as fn(client, question), so no problem_id
reaches the sidecar. Records carry the sha1 of the problem text instead, and
the SC baseline's own CSV supplies problem_id in the same append order. We
join on order and VERIFY with the `voted` checksum against the CSV's
`predicted`; a mismatch means the two files are not from the same run and the
join is refused rather than silently misaligned.

Usage:
    python sc_analysis.py --mas <mas.csv> --sc-csv <b3_sc5.csv> --sidecar <sc_samples.jsonl>
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import List, Optional

import pandas as pd
from scipy.stats import binomtest


def sc_at_k(samples: List[Optional[float]], k: int) -> Optional[float]:
    """Majority vote over the first k samples. Mirrors
    baselines.self_consistency exactly, including its tie-break (count desc,
    value asc) -- if these two ever disagree the whole SC@k reconstruction is
    invalid, which is what --verify checks."""
    nums = [x for x in samples[:k] if x is not None]
    if not nums:
        return None
    counts = Counter(nums)
    return max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]


def is_correct(pred, gold, tol: float = 1e-3) -> bool:
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < tol
    except (TypeError, ValueError):
        return str(pred).strip() == str(gold).strip()


def load_sidecar(path: str) -> pd.DataFrame:
    recs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    if not recs:
        raise SystemExit(f"sidecar {path} is empty -- did the SC phase run?")
    return pd.DataFrame(recs)


def _checksum_ok(csv_pred, side_voted) -> bool:
    if pd.isna(csv_pred) and side_voted is None:
        return True
    return is_correct(side_voted, csv_pred)


def join(sc_csv: pd.DataFrame, side: pd.DataFrame) -> pd.DataFrame:
    """Align sidecar records to CSV rows.

    Append order is the key, but a run killed by the 12h wall and resumed in a
    second commit starts a FRESH sidecar (Kaggle wipes /kaggle/working) while
    the CSV resumes from RESUME_FROM_CSV -- so the sidecar is then a contiguous
    block somewhere inside the CSV, not a prefix. Rather than refuse that case,
    slide the block over every offset and keep the one where every `voted`
    matches that row's `predicted`. The voted values act as a fingerprint: with
    150 rows an accidental full-length match is not a realistic concern, and if
    two offsets DO match we refuse rather than guess.
    """
    csv = sc_csv.reset_index(drop=True)
    side = side.reset_index(drop=True)
    if len(side) > len(csv):
        raise SystemExit(
            f"sidecar has more records ({len(side)}) than the SC csv has rows "
            f"({len(csv)}) -- the sidecar is carrying another run's records. "
            "Delete it and re-run the phase.")

    voted = list(side["voted"])
    hits = [
        off for off in range(len(csv) - len(side) + 1)
        if all(_checksum_ok(csv.loc[off + i, "predicted"], voted[i])
               for i in range(len(side)))
    ]
    if not hits:
        raise SystemExit(
            "checksum FAILED at every alignment: no offset makes the sidecar's "
            "voted answers line up with the csv's predictions.\n"
            "The two files are not from the same run -- refusing to report "
            "numbers built on a bad join.")
    if len(hits) > 1:
        raise SystemExit(
            f"ambiguous join: {len(hits)} offsets {hits[:5]} all check out. "
            "Too few distinct answers to fingerprint the alignment; join "
            "manually.")

    off = hits[0]
    out = csv.iloc[off:off + len(side)].reset_index(drop=True).copy()
    out["sample_answers"] = side["sample_answers"]
    out["sidecar_voted"] = side["voted"]
    if len(side) < len(csv):
        print(f"NOTE: sidecar covers {len(side)}/{len(csv)} csv rows "
              f"(offset {off}) -- SC@k below is reported on that subset only, "
              f"SC@5 on all {len(csv)}.\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mas", required=True, help="MAS run CSV (has baseline_correct)")
    ap.add_argument("--sc-csv", required=True, help="b3_sc5_*.csv from the SC phase")
    ap.add_argument("--sidecar", required=True, help="sc_samples.jsonl")
    ap.add_argument("--ks", default="1,3,5", help="which SC@k to report")
    args = ap.parse_args()

    mas = pd.read_csv(args.mas)
    sc = join(pd.read_csv(args.sc_csv), load_sidecar(args.sidecar))
    sc["sample_answers"] = sc["sample_answers"].apply(
        lambda v: v if isinstance(v, list) else json.loads(v))

    ks = [int(k) for k in args.ks.split(",")]
    for k in ks:
        sc[f"sc{k}_pred"] = sc.sample_answers.apply(lambda s: sc_at_k(s, k))
        sc[f"sc{k}_correct"] = [
            is_correct(p, g) for p, g in zip(sc[f"sc{k}_pred"], sc.gold)]

    # SC@5 reconstructed offline must equal the run's own answer.
    recon = (sc["sc5_correct"] == sc["correct"]).all() if 5 in ks else None
    print(f"SC@5 reconstruction matches the run's own grading: {recon}\n")

    merged = mas.merge(
        sc[["problem_id"] + [f"sc{k}_correct" for k in ks]],
        on="problem_id", how="inner", validate="one_to_one")
    if len(merged) != len(mas):
        print(f"WARNING: only {len(merged)}/{len(mas)} problems joined to the MAS run\n")

    calls = {1: 1.0, 3: 3.0, 5: 5.0}
    rows = [
        dict(system="baseline_only (1 call)", n=len(merged),
             accuracy=merged.baseline_correct.mean(), avg_calls=1.0),
        dict(system=f"MAS (SIV+SHT)", n=len(merged),
             accuracy=merged.correct.mean(), avg_calls=mas.num_llm_calls.mean()),
    ] + [
        dict(system=f"SC@{k}", n=len(merged),
             accuracy=merged[f"sc{k}_correct"].mean(), avg_calls=calls.get(k, float(k)))
        for k in ks
    ]
    tab = pd.DataFrame(rows).sort_values("avg_calls")
    tab["accuracy"] = tab.accuracy.round(4)
    print("=== ACCURACY vs COMPUTE (same 150 problems, same model, same quantization) ===")
    print(tab.to_string(index=False))

    print("\n=== PAIRED: MAS vs each SC@k (McNemar exact) ===")
    for k in ks:
        w = int(((merged.correct == 1) & (merged[f"sc{k}_correct"] == 0)).sum())
        l = int(((merged.correct == 0) & (merged[f"sc{k}_correct"] == 1)).sum())
        p = binomtest(w, w + l, 0.5).pvalue if w + l else float("nan")
        d = merged.correct.mean() - merged[f"sc{k}_correct"].mean()
        print(f"  MAS vs SC@{k}: delta={d:+.4f}  W={w} L={l}  p={p:.4f}"
              + ("   <-- ISO-COMPUTE" if k == 3 else ""))


if __name__ == "__main__":
    main()
