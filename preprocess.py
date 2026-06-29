"""Convert an arbitrary user CSV into a canonical CSV for the ``custom`` loader.

This is a thin, dependency-light helper (just pandas) that maps your columns onto
the canonical schema the framework expects. After running it you can load the
result with::

    from jss.datasets import load
    eval_df, demo_df = load("custom", mode="completion",
                            data_path="canonical.csv")

Required canonical schema
-------------------------
The output CSV always contains these columns:

    prompt_id  : str    unique id per row (auto-generated if you don't map one)
    prompt     : str    the user request                 (REQUIRED input column)
    completion : str    the model response               (required for mode="completion")
    category   : str    safety category label            (REQUIRED input column)
    severity   : float  1=high / 0=low / blank=benign    (REQUIRED input column)
    context    : str    short note on niche/novel terms  (optional; "" if absent)
    language   : str    language metadata                (optional; "" if absent)

Five-category vocabulary
------------------------
For the default ``custom`` family (``five_category``), ``category`` must be one
of:

    misinformation, self_harm, sexual_content, csam, violence_and_hate, none

Use ``--category-map`` (CLI) or ``category_map=`` (API) to translate your raw
labels onto this vocabulary. Anything blank / unmapped is treated as ``none``.

Severity convention
-------------------
    severity = 1   -> high severity (a real safety concern is present)
    severity = 0   -> low severity / neutral-but-on-topic
    severity blank -> benign: no safety category mentioned (becomes NaN)

Hold-out requirement
--------------------
The framework holds out 4 rows as in-context demonstrations (stratified into a
mix of safe/unsafe under the base definition). Your dataset must therefore have
MORE THAN 4 rows and contain a mix of safe and unsafe examples, otherwise the
loader will raise when it tries to build a balanced demonstration set.

CLI usage
---------
    python preprocess.py --in raw.csv --out canonical.csv \
        --prompt-col question --category-col harm_type --severity-col sev \
        [--completion-col answer] [--context-col notes] \
        [--language-col lang] [--id-col uid] \
        [--category-map '{"violence":"violence_and_hate","misinfo":"misinformation"}']
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

# Output column order for the canonical CSV.
CANONICAL_COLUMNS = [
    "prompt_id",
    "prompt",
    "completion",
    "category",
    "severity",
    "context",
    "language",
]

# Canonical five-category vocabulary (kept local so this script has no internal
# import dependency and runs standalone).
FIVE_CATEGORIES = [
    "misinformation",
    "self_harm",
    "sexual_content",
    "csam",
    "violence_and_hate",
    "none",
]


def to_canonical(
    df: pd.DataFrame,
    *,
    prompt_col: str,
    category_col: str,
    severity_col: str,
    completion_col: str | None = None,
    context_col: str | None = None,
    language_col: str | None = None,
    id_col: str | None = None,
    category_map: dict | None = None,
) -> pd.DataFrame:
    """Map an arbitrary frame onto the canonical columns.

    Parameters
    ----------
    df : pandas.DataFrame
        The raw input frame.
    prompt_col, category_col, severity_col : str
        Names of the required source columns.
    completion_col, context_col, language_col, id_col : str, optional
        Names of optional source columns. Missing ones default to "" (or an
        auto-generated id for ``prompt_id``).
    category_map : dict, optional
        Mapping from raw category labels to canonical ones. Applied
        case-insensitively; blanks / unmapped-and-not-already-canonical values
        fall back to "none".

    Returns
    -------
    pandas.DataFrame with exactly :data:`CANONICAL_COLUMNS`.
    """
    for required, col in [
        ("prompt", prompt_col),
        ("category", category_col),
        ("severity", severity_col),
    ]:
        if col not in df.columns:
            raise ValueError(
                f"--{required}-col {col!r} not found in input columns: "
                f"{list(df.columns)}"
            )

    n = len(df)
    out = pd.DataFrame(index=range(n))

    # prompt_id
    if id_col and id_col in df.columns:
        out["prompt_id"] = df[id_col].astype(str).values
    else:
        out["prompt_id"] = [f"row-{i:05d}" for i in range(n)]

    # prompt (required)
    out["prompt"] = df[prompt_col].fillna("").astype(str).values

    # completion (optional)
    if completion_col and completion_col in df.columns:
        out["completion"] = df[completion_col].fillna("").astype(str).values
    else:
        out["completion"] = ""

    # category (required) with optional mapping
    out["category"] = [
        _map_category(v, category_map) for v in df[category_col].values
    ]

    # severity (required) -> float, blanks stay NaN
    out["severity"] = pd.to_numeric(df[severity_col], errors="coerce").values

    # context (optional)
    if context_col and context_col in df.columns:
        out["context"] = df[context_col].fillna("").astype(str).values
    else:
        out["context"] = ""

    # language (optional)
    if language_col and language_col in df.columns:
        out["language"] = df[language_col].fillna("").astype(str).values
    else:
        out["language"] = ""

    return out[CANONICAL_COLUMNS]


def _map_category(raw, category_map: dict | None) -> str:
    """Translate a raw category onto the canonical vocabulary.

    Blanks / NaN -> "none". A user ``category_map`` is applied case-insensitively;
    values already in the canonical vocabulary pass through; anything else is
    kept unchanged (so the loader's validation can surface unexpected labels).
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return "none"
    s = str(raw).strip()
    if s == "" or s.lower() in ("nan", "none", "null"):
        return "none"

    if category_map:
        # Case-insensitive lookup.
        lower_map = {str(k).lower(): v for k, v in category_map.items()}
        if s.lower() in lower_map:
            return lower_map[s.lower()]

    if s.lower() in FIVE_CATEGORIES:
        return s.lower()
    return s


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert a raw CSV into a canonical CSV for the 'custom' "
        "dataset loader.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Canonical categories: " + ", ".join(FIVE_CATEGORIES),
    )
    p.add_argument("--in", dest="in_path", required=True, help="Input CSV path")
    p.add_argument("--out", dest="out_path", required=True, help="Output CSV path")
    p.add_argument("--prompt-col", required=True, help="Source column for prompt")
    p.add_argument(
        "--category-col", required=True, help="Source column for category"
    )
    p.add_argument(
        "--severity-col", required=True, help="Source column for severity"
    )
    p.add_argument("--completion-col", default=None, help="Source column for completion")
    p.add_argument("--context-col", default=None, help="Source column for context")
    p.add_argument("--language-col", default=None, help="Source column for language")
    p.add_argument("--id-col", default=None, help="Source column for prompt_id")
    p.add_argument(
        "--category-map",
        default=None,
        help='JSON object mapping raw labels to canonical ones, e.g. '
        '\'{"violence":"violence_and_hate"}\'',
    )
    return p


def main(argv=None) -> None:
    args = _build_arg_parser().parse_args(argv)

    category_map = json.loads(args.category_map) if args.category_map else None

    df = pd.read_csv(args.in_path)
    canonical = to_canonical(
        df,
        prompt_col=args.prompt_col,
        category_col=args.category_col,
        severity_col=args.severity_col,
        completion_col=args.completion_col,
        context_col=args.context_col,
        language_col=args.language_col,
        id_col=args.id_col,
        category_map=category_map,
    )
    canonical.to_csv(args.out_path, index=False)

    n = len(canonical)
    print(f"Wrote {n} rows to {args.out_path}")
    if n <= 4:
        print(
            "WARNING: the framework holds out 4 demonstration rows, so a usable "
            "dataset needs MORE THAN 4 rows with a mix of safe/unsafe examples."
        )


if __name__ == "__main__":
    main()
