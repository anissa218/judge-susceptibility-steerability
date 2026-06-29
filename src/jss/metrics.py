"""Turn a long predictions frame into per-condition summary metrics.

Pipeline: apply the NaN policy -> per-seed accuracy/F1 -> mean/std across seeds
-> majority-vote flip-rate vs each experiment's base condition -> (for
steerability) the *expected* flip-rate implied by the gold-label change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from . import extract
from .experiments import base_condition_for

GROUP = ["judge", "experiment", "condition"]


def _majority(pred_by_seed: pd.DataFrame) -> pd.Series:
    """Majority vote per prompt_id across seeds (mean >= 0.5 -> 1).

    With an even number of seeds a 50/50 tie resolves to 1 (unsafe); use an odd
    --seeds (default 5) to avoid ties.
    """
    m = pred_by_seed.groupby("prompt_id")["prediction"].mean()
    return (m >= 0.5).astype(int)


def summarize(pred_df: pd.DataFrame, family: str | None = None) -> pd.DataFrame:
    """Compute the summary table from a long predictions frame."""
    if pred_df.empty:
        return pd.DataFrame()
    if family is None:
        family = pred_df["family"].iloc[0] if "family" in pred_df.columns else None
    if family is None:
        raise ValueError("family unknown: pass family= or include a 'family' column")

    clean, _report = extract.apply_nan_policy(pred_df, GROUP)
    clean = clean.copy()
    clean["prediction"] = pd.to_numeric(clean["prediction"], errors="coerce")
    clean["gold"] = pd.to_numeric(clean["gold"], errors="coerce")

    # Majority vote + gold per (judge, experiment, condition).
    maj: dict[tuple, pd.Series] = {}
    gold_by_pid: dict[tuple, pd.Series] = {}
    for (judge, exp, cond), g in clean.groupby(GROUP):
        maj[(judge, exp, cond)] = _majority(g)
        gold_by_pid[(judge, exp, cond)] = g.groupby("prompt_id")["gold"].first()

    rows = []
    meta_cols = ["definition", "task", "n_demos", "misleading", "with_context"]
    for (judge, exp, cond), g in clean.groupby(GROUP):
        # Per-seed accuracy / F1 (NaNs already filled/excluded by policy).
        accs, f1s = [], []
        for _seed, gs in g.groupby("seed"):
            gs = gs.dropna(subset=["prediction", "gold"])
            if gs.empty:
                continue
            y_true = gs["gold"].astype(int)
            y_pred = gs["prediction"].astype(int)
            accs.append(accuracy_score(y_true, y_pred))
            f1s.append(f1_score(y_true, y_pred, zero_division=0))

        base_cond = base_condition_for(exp, family)
        # Flip-rate vs base (majority vote over shared prompt_ids).
        flip = np.nan
        if (judge, exp, base_cond) in maj and cond != base_cond:
            a = maj[(judge, exp, cond)]
            b = maj[(judge, exp, base_cond)]
            shared = a.index.intersection(b.index)
            if len(shared):
                flip = float((a.loc[shared] != b.loc[shared]).mean())
        elif cond == base_cond:
            flip = 0.0

        # Expected flip-rate (steerability only): how much GT *should* move to adapt to the new policy.
        exp_flip = np.nan
        if exp.startswith("steer") and (judge, exp, base_cond) in gold_by_pid:
            ga = gold_by_pid[(judge, exp, cond)]
            gb = gold_by_pid[(judge, exp, base_cond)]
            shared = ga.index.intersection(gb.index)
            if len(shared):
                exp_flip = float((ga.loc[shared] != gb.loc[shared]).mean())

        meta = {c: (g[c].iloc[0] if c in g.columns else None) for c in meta_cols}
        rows.append(
            {
                "dataset": g["dataset"].iloc[0] if "dataset" in g.columns else None,
                "mode": g["mode"].iloc[0] if "mode" in g.columns else None,
                "judge": judge,
                "experiment": exp,
                "condition": cond,
                **meta,
                "base_condition": base_cond,
                "n_samples": int(g["prompt_id"].nunique()),
                "accuracy_mean": float(np.mean(accs)) if accs else np.nan,
                "accuracy_std": float(np.std(accs, ddof=0)) if accs else np.nan,
                "f1_mean": float(np.mean(f1s)) if f1s else np.nan,
                "f1_std": float(np.std(f1s, ddof=0)) if f1s else np.nan,
                "flip_rate": flip,
                "expected_flip_rate": exp_flip,
            }
        )

    return pd.DataFrame(rows)


def summarize_dir(results_subdir: str) -> pd.DataFrame:
    """Read ``<results_subdir>/predictions.csv``, write+return summary.csv."""
    from pathlib import Path

    sub = Path(results_subdir)
    pred = pd.read_csv(sub / "predictions.csv")
    summary = summarize(pred)
    summary.to_csv(sub / "summary.csv", index=False)
    return summary
