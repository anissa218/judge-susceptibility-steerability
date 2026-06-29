"""Canonical data schema + validation + demonstration hold-out.

Every dataset loader (see ``datasets.py``) and any custom CSV (see
``preprocess.py``) must produce a DataFrame with the canonical columns below.
Downstream modules (definitions, templates, experiments, metrics) only ever see
this canonical frame, so they stay dataset-agnostic.

Canonical columns
-----------------
prompt_id : str    unique row id
prompt    : str    the user request
completion: str    the model response ("" when judging prompts only)
category  : str    family-specific category label (see families below); for the
                   five-category family one of FIVE_CATEGORIES (incl. "none")
severity  : float  1.0 = high severity, 0.0 = low / neutral-but-on-topic,
                   NaN = benign / no safety category mentioned
context   : str    short explanation of niche/novel terms ("" if none)
language  : str    optional metadata ("" if unknown)

Some families attach extra columns the canonical contract tolerates:
  - sorry_bench:  sb_category_id (int 1..44)
  - sports:       stratum ("ball" | "nonball" | "unrelated")

The ``family`` of a dataset (which safety-definition set applies) is carried on
``df.attrs["family"]`` and is one of definitions.FAMILIES.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- Canonical categories for the five-category family -----------------------
FIVE_CATEGORIES = [
    "misinformation",
    "self_harm",
    "sexual_content",
    "csam",
    "violence_and_hate",
    "none",
]

CANONICAL_COLUMNS = [
    "prompt_id",
    "prompt",
    "completion",
    "category",
    "severity",
    "context",
    "language",
]


def standardise(df: pd.DataFrame, family: str) -> pd.DataFrame:
    """Coerce a loader's frame to the canonical schema and add ``family``.

    Missing optional columns are filled with sensible defaults; ``prompt_id`` is
    generated when absent. Extra family columns are preserved.
    """
    df = df.copy()

    if "prompt_id" not in df.columns:
        df["prompt_id"] = [f"row-{i:05d}" for i in range(len(df))]
    df["prompt_id"] = df["prompt_id"].astype(str)

    if "prompt" not in df.columns:
        raise ValueError("Canonical frame requires a 'prompt' column")
    df["prompt"] = df["prompt"].fillna("").astype(str)

    for col, default in [("completion", ""), ("context", ""), ("language", "")]:
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna("").astype(str)

    if "category" not in df.columns:
        raise ValueError("Canonical frame requires a 'category' column")
    # Treat explicit nulls / empty strings as the 'none' category.
    df["category"] = (
        df["category"].astype("object").where(df["category"].notna(), "none")
    )
    df["category"] = df["category"].replace({"": "none", "None": "none", "nan": "none"})

    if "severity" not in df.columns:
        raise ValueError("Canonical frame requires a 'severity' column")
    df["severity"] = pd.to_numeric(df["severity"], errors="coerce")  # keeps NaN

    df.attrs["family"] = family
    return df


def validate(df: pd.DataFrame, mode: str) -> None:
    """Light sanity checks. ``mode`` is 'prompt' or 'completion'."""
    if mode not in ("prompt", "completion"):
        raise ValueError(f"mode must be 'prompt' or 'completion', got {mode!r}")
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Canonical frame missing columns: {missing}")
    if df["prompt_id"].duplicated().any():
        raise ValueError("prompt_id values must be unique")
    if mode == "completion" and (df["completion"].str.len() == 0).all():
        raise ValueError(
            "mode='completion' but no rows have a completion. Use mode='prompt' "
            "or provide a 'completion'/'response' column."
        )


def holdout_demos(
    df: pd.DataFrame,
    base_gold,
    n: int = 4,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split off ``n`` demonstration rows, stratified by safe/unsafe.

    Returns ``(eval_df, demo_df)``. The demo rows are removed from eval_df.

    Raises error if the dataset is too small to hold out ``n`` rows.
    """
    if len(df) <= n:
        raise ValueError(
            f"Dataset has {len(df)} rows; need > {n} to hold out {n} demonstrations."
        )
    labels = df.apply(lambda r: base_gold(r), axis=1)
    rng = np.random.default_rng(seed)

    unsafe_idx = list(df.index[labels == 1])
    safe_idx = list(df.index[labels == 0])
    rng.shuffle(unsafe_idx)
    rng.shuffle(safe_idx)

    want_unsafe = n // 2
    want_safe = n - want_unsafe
    chosen = unsafe_idx[:want_unsafe] + safe_idx[:want_safe]

    # Backfill if one class is short.
    if len(chosen) < n:
        pool = [i for i in (unsafe_idx + safe_idx) if i not in chosen]
        chosen += pool[: n - len(chosen)]

    demo_df = df.loc[chosen].copy()
    eval_df = df.drop(index=chosen).reset_index(drop=True)
    demo_df = demo_df.reset_index(drop=True)
    demo_df.attrs["family"] = df.attrs.get("family")
    eval_df.attrs["family"] = df.attrs.get("family")
    return eval_df, demo_df
