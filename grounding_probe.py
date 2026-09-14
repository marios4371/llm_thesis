"""
[v15.9] Feasibility probe: can a symbolic check DETECT the translation errors?

85% of the system's errors are translation-layer -- the model misreads the
problem and then does correct algebra on wrong premises. SIV cannot see them
because it verifies that the blueprint is consistent WITH ITSELF, not that it
matches the problem: on the 27 errors of the v15.6 run it reports "verified"
20 times, and the matched/total givens ratio has median 1.00 on errors and on
correct rows alike.

This asks whether grounding the blueprint's givens against the PROBLEM TEXT
separates them. A given whose value never appears in the text is suspicious --
but not automatically wrong, because some givens are legitimately derived
rather than quoted. So the number that matters is not how often ungrounded
givens occur, it is whether they occur MORE on the rows the system gets wrong.
A detector that fires equally on both is worthless however intuitive it looks,
which is the same trap that sank the selector rules in Run 0.

Offline: problem ids are deterministic dataset indices (gsm-hard_1298 is row
1298), so the text is recoverable from the HF cache with no GPU and no re-draw.
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

RUN = 'results_September/results/MAS_SHT/results/mas_sht_math7b_20260912_094311.csv'
NUM_RE = re.compile(r'-?\d[\d,]*\.?\d*')


def load_texts() -> Dict[str, str]:
    """problem_id -> problem text, straight from the cached datasets."""
    from datasets import load_dataset
    texts: Dict[str, str] = {}

    ds = load_dataset("reasoning-machines/gsm-hard", split="train")
    for i in range(len(ds)):
        texts[f"gsm-hard_{i}"] = ds[i].get("input", "")

    ds = load_dataset("openai/gsm8k", "main", split="test")
    for i in range(len(ds)):
        texts[f"gsm8k_test_{i}"] = ds[i].get("question", "")

    ds = load_dataset("ChilleD/SVAMP", split="test")
    for i in range(len(ds)):
        q = ds[i].get("question_concat", None)
        if not q:
            q = (str(ds[i].get("Body", "")).strip() + "\n"
                 + str(ds[i].get("Question", "")).strip()).strip()
        texts[f"svamp_test_{i}"] = q
    return texts


def text_numbers(text: str) -> List[float]:
    out = []
    for m in NUM_RE.findall(text or ""):
        try:
            out.append(round(float(m.replace(",", "")), 6))
        except ValueError:
            pass
    return out


def grounded(val: float, nums: List[float]) -> bool:
    """A given counts as grounded if the text states that number outright, or
    states it scaled by a power of ten (percent/unit shifts are transcription,
    not invention)."""
    for n in nums:
        if abs(val - n) < 1e-6:
            return True
        for s in (0.01, 0.1, 10.0, 100.0):
            if abs(val - n * s) < 1e-6:
                return True
    return False


def analyse() -> pd.DataFrame:
    d = pd.read_csv(RUN)
    texts = load_texts()
    rows = []
    for _, r in d.iterrows():
        txt = texts.get(r.problem_id)
        try:
            givens = json.loads(r.blueprint_givens) if isinstance(r.blueprint_givens, str) else {}
        except (json.JSONDecodeError, TypeError):
            givens = {}
        vals = {k: v for k, v in givens.items() if isinstance(v, (int, float))}
        if txt is None or not vals:
            rows.append(dict(problem_id=r.problem_id, correct=bool(r.correct),
                             n_givens=len(vals), n_ungrounded=np.nan,
                             have_text=txt is not None))
            continue
        nums = text_numbers(txt)
        ung = [k for k, v in vals.items() if not grounded(float(v), nums)]
        rows.append(dict(problem_id=r.problem_id, correct=bool(r.correct),
                         n_givens=len(vals), n_ungrounded=len(ung),
                         ungrounded_keys="; ".join(ung[:3]), have_text=True))
    return pd.DataFrame(rows)


def main() -> None:
    f = analyse()
    print(f"rows: {len(f)}   text recovered for {int(f.have_text.sum())}   "
          f"blueprints with numeric givens: {int(f.n_givens.gt(0).sum())}\n")
    u = f[f.n_ungrounded.notna()].copy()
    u['has_ung'] = u.n_ungrounded > 0

    print("=== Does an ungrounded given occur more often on the ERRORS? ===")
    t = u.groupby('correct').agg(
        rows=('has_ung', 'size'),
        pct_with_ungrounded=('has_ung', 'mean'),
        mean_givens=('n_givens', 'mean'),
        mean_ungrounded=('n_ungrounded', 'mean'))
    t.index = ['WRONG', 'RIGHT']
    print(t.round(4).to_string())

    tp = int((u.has_ung & ~u.correct).sum())
    fp = int((u.has_ung & u.correct).sum())
    fn = int((~u.has_ung & ~u.correct).sum())
    tn = int((~u.has_ung & u.correct).sum())
    base = float((~u.correct).mean())
    prec = tp / (tp + fp) if tp + fp else float('nan')
    print(f"\n=== 'has an ungrounded given' as an ERROR DETECTOR ===")
    print(f"  fires on {tp+fp} rows: {tp} are genuinely wrong, {fp} are fine")
    print(f"  precision {prec:.3f}   vs base error rate {base:.3f}   "
          f"-> lift {prec-base:+.3f}")
    print(f"  recall {tp/(tp+fn) if tp+fn else float('nan'):.3f}  "
          f"(catches {tp} of the {tp+fn} errors)")

    print("\n=== examples: WRONG rows carrying an ungrounded given ===")
    ex = u[(~u.correct) & u.has_ung].head(8)
    for _, r in ex.iterrows():
        print(f"  {r.problem_id:18s} {int(r.n_ungrounded)}/{int(r.n_givens)} ungrounded"
              f"   {r.ungrounded_keys}")


if __name__ == '__main__':
    main()
