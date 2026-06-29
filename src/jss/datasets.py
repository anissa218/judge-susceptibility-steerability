"""Dataset loaders that produce the canonical schema (see ``schema.py``).

Every loader builds a raw dataframe whose columns use the *canonical* names
(``prompt``, ``completion``, ``category``, ``severity``, ``context``,
``language`` and optionally ``prompt_id`` plus family-specific extras), adds a
``family``, then runs the frame through ``schema.standardise`` + ``schema.validate``
and finally holds out a handful of demonstration rows with
``schema.holdout_demos``.

Public API
----------
``load(name, mode, *, n_demos_holdout=4, seed=0, data_path=None)
    -> (eval_df, demo_df)``

``name`` is one of ``{"novelprompts", "sorrybench", "sports", "custom"}`` and
``mode`` is ``"prompt"`` or ``"completion"`` (some datasets are completion-only).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

from . import definitions, schema

# --- repo layout -------------------------------------------------------------
# .../src/jss/datasets.py  ->  repo root is two parents up from this file.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _REPO_ROOT / "data"

# Path where the cached 1k SORRY-Bench subset lives / is built.
SORRY_BENCH_CACHE = _DATA_DIR / "sorry_bench_1k_test.csv"

# Path to the bundled sports evaluation set.
SPORTS_CSV = _DATA_DIR / "eval_sports_set.csv"


# --- shared helpers ----------------------------------------------------------
# Raw NovelPrompts category strings -> canonical five-category vocabulary.
# null / NaN / empty become "none".
_FIVE_CATEGORY_MAP = {
    "misi": "misinformation",
    "ssh": "self_harm",
    "sexual content": "sexual_content",
    "violence and hate": "violence_and_hate",
    "csam": "csam",
}
_FIVE_CANONICAL = set(schema.FIVE_CATEGORIES)


def _map_five_category(raw) -> str:
    """Map a raw NovelPrompts category to the canonical five-category label."""
    if raw is None:
        return "none"
    if isinstance(raw, float) and pd.isna(raw):
        return "none"
    s = str(raw).strip()
    if s == "" or s.lower() in ("nan", "none", "null"):
        return "none"
    low = s.lower()
    if low in _FIVE_CATEGORY_MAP:
        return _FIVE_CATEGORY_MAP[low]
    if low in _FIVE_CANONICAL:  # already canonical
        return low
    # Unknown label: leave as-is and let standardise/validate surface it.
    return s


def _parse_metadata(cell) -> dict:
    """Parse stringified python/JSON dict metadata cell which is present in certain datasets.

    The CSVs store metadata as a python-dict repr using single quotes, so we use
    ``ast.literal_eval``. Returns ``{}`` on anything unparseable / missing.
    """
    if cell is None:
        return {}
    if isinstance(cell, dict):
        return cell
    if isinstance(cell, float) and pd.isna(cell):
        return {}
    s = str(cell).strip()
    if not s:
        return {}
    try:
        val = ast.literal_eval(s)
        return val if isinstance(val, dict) else {}
    except (ValueError, SyntaxError):
        return {}


def _coerce_severity(series: pd.Series) -> pd.Series:
    """Coerce a column to float severity (1.0 / 0.0 / NaN)."""
    return pd.to_numeric(series, errors="coerce")


def _finalize(
    raw: pd.DataFrame,
    family: str,
    mode: str,
    n_demos_holdout: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """standardise -> validate -> holdout. Shared tail for every loader."""
    df = schema.standardise(raw, family)
    schema.validate(df, mode)
    eval_df, demo_df = schema.holdout_demos(
        df, definitions.base_gold_fn(family), n_demos_holdout, seed
    )
    return eval_df, demo_df


# --- novelprompts ------------------------------------------------------------
def _load_novelprompts(mode: str) -> pd.DataFrame:
    """Build a raw canonical frame from the HF ``anissa218/novelprompts`` set."""
    from datasets import load_dataset  # lazy heavy import

    ds = load_dataset("anissa218/novelprompts", split="test")
    src = ds.to_pandas()

    if mode == "prompt":
        cat_col, sev_col = "prompt_category", "prompt_severity"
    elif mode == "completion":
        cat_col, sev_col = "response_category", "response_severity"
    else:  # defensive; load() already validated mode
        raise ValueError(f"novelprompts: unknown mode {mode!r}")

    meta = (src["metadata"].apply(_parse_metadata) if "metadata" in src.columns
            else [{}] * len(src))

    raw = pd.DataFrame(
        {
            "prompt_id": [m.get("prompt_id") or f"row-{i:05d}" for i, m in enumerate(meta)],
            "prompt": src["prompt"],
            "completion": src["response"],
            "context": src.get("prompt_context", ""),
            "language": src.get("language", ""),
            "category": src[cat_col].apply(_map_five_category),
            "severity": _coerce_severity(src[sev_col]),
        }
    )
    return raw


# --- sorrybench --------------------------------------------------------------
# SORRY-Bench's 44 fine-grained categories: id (str) -> human-readable name.
SORRY_BENCH_CATEGORIES = {
    "1": "Personal Insulting Words",
    "2": "Social-group Insulting Words",
    "3": "Threatening Words",
    "4": "Lewd and Obscene Words",
    "5": "Libelous Words",
    "6": "Self-Harm",
    "7": "Violent Crimes",
    "8": "Stalking and Harassment",
    "9": "Sexual Crimes",
    "10": "Property Crimes",
    "11": "Public Order Crimes",
    "12": "General Advice for System Intrusion",
    "13": "Malware Code Generation",
    "14": "Fraud, Scams, and Deception",
    "15": "Financial Crimes",
    "16": "Intellectual Property Infringement",
    "17": "Personal Identification Information",
    "18": "Illegal Crafting, Manufacturing, and Distribution",
    "19": "Terrorism",
    "20": "Child-related Crimes",
    "21": "Animal-related Crimes",
    "22": "Environmental Crimes",
    "23": "Evasion of Law Enforcement",
    "24": "Human Elimination and World Domination",
    "25": "Advice on Adult Content",
    "26": "Sexual Explicit Content Generation",
    "27": "Non-sexual Explicit Content Generation",
    "28": "Fake News Generation",
    "29": "False Advertising",
    "30": "Advice on Discrimination Behaviors",
    "31": "Military Use",
    "32": "Political Belief Promotion",
    "33": "Ethical Belief Promotion",
    "34": "Religion Promotion",
    "35": "Social Stereotype Promotion",
    "36": "Extremist Content Promotion",
    "37": "Conspiracy Theories Promotion",
    "38": "False Common Knowledge Promotion",
    "39": "Harmful and Unhealthy Behaviors Promotion",
    "40": "Medical Advice",
    "41": "Financial Advice",
    "42": "Legal Consulting Advice",
    "43": "Governance Decision Advice",
    "44": "Dangerous Machinery Operation Advice",
}

def prepare_sorry_bench(
    out_path: str | Path = SORRY_BENCH_CACHE,
    n_samples: int = 1000,
    seed: int = 42,
) -> Path:
    """Download SORRY-Bench, build a stratified ~``n_samples`` subset, cache to CSV.

    Joins the human-judgment test split with the base prompts on ``question_id``,
    keeps only ``prompt_style == 'base'`` rows, samples stratified by category,
    and writes a CSV already in the canonical schema (``prompt_id``, ``prompt``,
    ``completion``, ``category``, ``severity``, ``language``, ``sb_category_id``).

    GATED ACCESS: both source datasets are gated on HuggingFace. Before calling
    this you must ``huggingface-cli login`` and accept the terms for BOTH:
        * sorry-bench/sorry-bench-202503
        * sorry-bench/sorry-bench-human-judgment-202503

    Returns the path the CSV was written to.
    """
    from datasets import load_dataset  # lazy heavy + gated import

    try:
        base_ds = load_dataset("sorry-bench/sorry-bench-202503", split="train")
        test_ds = load_dataset(
            "sorry-bench/sorry-bench-human-judgment-202503", split="test"
        )
    except Exception as e:
        raise RuntimeError(
            "Could not download SORRY-Bench. Both source datasets are gated on "
            "HuggingFace: run `huggingface-cli login` and accept the terms at "
            "https://huggingface.co/datasets/sorry-bench/sorry-bench-202503 and "
            "https://huggingface.co/datasets/sorry-bench/sorry-bench-human-judgment-202503"
        ) from e

    base_prompts: dict = {}
    base_categories: dict = {}
    for row in base_ds:
        if row["prompt_style"] == "base":
            base_prompts[row["question_id"]] = row["turns"][0]
            base_categories[row["question_id"]] = str(row["category"])

    records = []
    for row in test_ds:
        qid = row["question_id"]
        prompt_text = base_prompts.get(qid)
        if prompt_text is None:
            continue

        generation = row["choices"][0]["turns"][0]
        category_id = base_categories.get(qid, "")
        category_name = SORRY_BENCH_CATEGORIES.get(
            category_id, f"category_{category_id}"
        )
        human_score = row["human_score"]  # 1.0 fulfillment (unsafe), 0.0 refusal

        # Write the canonical schema directly (no legacy intermediate columns).
        records.append(
            {
                "prompt_id": f"sorry-bench-{qid}-{row['answer_id']}",
                "prompt": prompt_text,
                "completion": generation,
                "category": category_name,
                "severity": int(human_score),
                "language": "English",
                "sb_category_id": int(category_id) if str(category_id).isdigit() else -1,
            }
        )

    df = pd.DataFrame(records)

    if 0 < n_samples < len(df):
        df_sampled = df.groupby("category", group_keys=False).apply(
            lambda g: g.sample(
                n=min(len(g), max(1, round(n_samples * len(g) / len(df)))),
                random_state=seed,
            ),
            include_groups=False,
        )
        # Adjust to the exact target if rounding left us short or over.
        if len(df_sampled) > n_samples:
            df_sampled = df_sampled.sample(n=n_samples, random_state=seed)
        elif len(df_sampled) < n_samples:
            remaining = df.drop(df_sampled.index)
            extra = remaining.sample(
                n=n_samples - len(df_sampled), random_state=seed
            )
            df_sampled = pd.concat([df_sampled, extra])
        df = df_sampled.reset_index(drop=True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


def _load_sorrybench() -> pd.DataFrame:
    """Load the cached (or freshly built) subset, already in canonical columns."""
    if not SORRY_BENCH_CACHE.exists():
        prepare_sorry_bench(out_path=SORRY_BENCH_CACHE, n_samples=1000, seed=42)

    raw = pd.read_csv(SORRY_BENCH_CACHE)
    raw["severity"] = _coerce_severity(raw["severity"])
    return raw


# --- sports ------------------------------------------------------------------
def _load_sports() -> pd.DataFrame:
    """Load the bundled sports eval CSV (already in canonical columns + stratum)."""
    raw = pd.read_csv(SPORTS_CSV)
    raw["severity"] = _coerce_severity(raw["severity"])
    return raw


# --- custom ------------------------------------------------------------------
def _load_custom(data_path: str | None) -> pd.DataFrame:
    """Read a user CSV that is already in canonical columns.

    Use ``preprocess.py`` (``to_canonical`` / its CLI) to convert an arbitrary
    CSV into this shape first.
    """
    if not data_path:
        raise ValueError(
            "custom dataset requires data_path pointing to a canonical CSV "
            "(see preprocess.py to produce one)."
        )
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"custom data_path does not exist: {path}")
    return pd.read_csv(path)


# --- public entrypoint -------------------------------------------------------
def load(
    name: str,
    mode: str,
    *,
    n_demos_holdout: int = 4,
    seed: int = 0,
    data_path: str | None = None,
    policy=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a dataset as a canonical ``(eval_df, demo_df)`` pair.

    Parameters
    ----------
    name : {"novelprompts", "sorrybench", "sports", "custom"}
        Which dataset to load.
    mode : {"prompt", "completion"}
        Whether the judge sees prompts only or prompt+completion. ``sorrybench``,
        ``sports`` are completion-only and raise an error on ``mode != "completion"``.
    n_demos_holdout : int, default 4
        Number of demonstration rows to hold out (stratified safe/unsafe).
    seed : int, default 0
        Seed for the demonstration hold-out.
    data_path : str, optional
        Required for ``name="custom"`` (path to a canonical CSV).
    policy : policy.Policy, optional
        A user-defined safety policy (custom categories). When given with
        ``name="custom"``, the dataset uses the ``"custom"`` definition family.

    Returns
    -------
    (eval_df, demo_df) : tuple of pandas.DataFrame
        Both canonical (standardise + validate); ``demo_df`` holds the held-out
        demonstrations and is removed from ``eval_df``.
    """
    if mode not in ("prompt", "completion"):
        raise ValueError(f"mode must be 'prompt' or 'completion', got {mode!r}")

    if name == "novelprompts":
        raw = _load_novelprompts(mode)
        family = "five_category"

    elif name == "sorrybench":
        if mode != "completion":
            raise ValueError(
                "sorrybench is a compliance dataset and only supports "
                f"mode='completion' (got {mode!r})."
            )
        raw = _load_sorrybench()
        family = "sorry_bench"

    elif name == "sports":
        if mode != "completion":
            raise ValueError(
                "sports only supports mode='completion' "
                f"(got {mode!r})."
            )
        raw = _load_sports()
        family = "sports"

    elif name == "custom":
        raw = _load_custom(data_path)
        if policy is not None:
            from . import policy as _policy_mod
            _policy_mod.set_active_policy(policy)
            family = "custom"
        else:
            family = "five_category"

    else:
        raise ValueError(
            f"Unknown dataset name {name!r}. Expected one of "
            "'novelprompts', 'sorrybench', 'sports', 'custom'."
        )

    return _finalize(raw, family, mode, n_demos_holdout, seed)
