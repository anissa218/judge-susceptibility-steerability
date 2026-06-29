"""Experiment matrix: the set of judge-call conditions for a dataset.

Each ``Condition`` fully specifies how to render a sample and how to score it
(which definition GT to use). The runner iterates conditions x seeds x
samples x judges.

Experiments
-----------
susc_context  (susceptibility to novel context; five_category w/ context only)
    context_none  (base) | context_correct | context_shuffled
susc_demos    (susceptibility to demonstrations; all families)
    demos_0 (base) | demos_{1,2,4}_correct | demos_{1,2,4}_misleading
steer_safety  (steerability via safety-policy edits)
    def_<name> for STEERABILITY_DEFINITIONS[family] (+ def_none for five_category)
steer_classification (same definitions, but reframed as A/B classification)
    cls_<name>
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import definitions as D

DEMO_COUNTS = [1, 2, 4]


@dataclass
class Condition:
    experiment: str            # susc_context | susc_demos | steer_safety | steer_classification
    condition: str             # unique human-readable label within the dataset
    task: str                  # "safety" | "classification"
    definition: str            # definitions name used for gold + template
    with_context: bool = False
    context_mode: str = "none" # none | correct | shuffled
    n_demos: int = 0
    misleading: bool = False
    demos: list[dict] = field(default_factory=list)


# Which classification definitions each family exposes (templates exist for these).
_CLS_DEFS = {
    "five_category": ["base", "strict", "swapped"],
    "sorry_bench": ["explicit", "definition_a", "definition_b"],
    "sports": [],  # no classification framing for the sports toy task
    "custom": ["base", "strict", "swapped"],
}

# Families whose policy is described over (category, severity) and which support
# the context experiment and the no-policy ablation.
_CATEGORY_FAMILIES = ("five_category", "custom")


def _demo_records(demo_df: pd.DataFrame, family: str, n: int) -> list[dict]:
    """First ``n`` held-out rows as {prompt, completion, label} with base gold."""
    base_def = D.BASE_DEFINITION[family]
    recs = []
    for _, row in demo_df.head(n).iterrows():
        recs.append(
            {
                "prompt": row["prompt"],
                "completion": row.get("completion", ""),
                "label": D.gold(row, family, base_def),
            }
        )
    return recs


def base_condition_for(experiment: str, family: str) -> str:
    """The reference condition each experiment's flip-rate is measured against."""
    if experiment == "susc_context":
        return "context_none"
    if experiment == "susc_demos":
        return "demos_0"
    if experiment == "steer_safety":
        return f"def_{D.BASE_DEFINITION[family]}"
    if experiment == "steer_classification":
        defs = _CLS_DEFS[family]
        if not defs:
            raise ValueError(f"Family {family!r} has no classification definitions.")
        return f"cls_{defs[0]}"
    raise ValueError(f"Unknown experiment {experiment!r}")


def build_conditions(
    eval_df: pd.DataFrame,
    demo_df: pd.DataFrame,
    family: str,
    mode: str,
    experiments: list[str],
) -> list[Condition]:
    """Construct the condition list for the requested experiment groups.

    ``experiments`` may contain the umbrella names "susceptibility" /
    "steerability" or the fine-grained experiment names.
    """
    want = set(experiments)
    susc = {"susceptibility", "susc_context", "susc_demos"} & want
    steer = {"steerability", "steer_safety", "steer_classification"} & want

    conditions: list[Condition] = []
    base_def = D.BASE_DEFINITION[family]

    # --- susceptibility: context (five_category + actual context only) --------
    has_context = (eval_df["context"].fillna("").str.len() > 0).any()
    if (("susceptibility" in want) or ("susc_context" in want)) and family in _CATEGORY_FAMILIES and has_context:
        conditions.append(Condition("susc_context", "context_none", "safety", base_def, with_context=False, context_mode="none"))
        conditions.append(Condition("susc_context", "context_correct", "safety", base_def, with_context=True, context_mode="correct"))
        conditions.append(Condition("susc_context", "context_shuffled", "safety", base_def, with_context=True, context_mode="shuffled"))

    # --- susceptibility: demonstrations (all families) -----------------------
    if ("susceptibility" in want) or ("susc_demos" in want):
        conditions.append(Condition("susc_demos", "demos_0", "safety", base_def))
        for n in DEMO_COUNTS:
            recs = _demo_records(demo_df, family, n)
            conditions.append(Condition("susc_demos", f"demos_{n}_correct", "safety", base_def, n_demos=n, misleading=False, demos=recs))
            conditions.append(Condition("susc_demos", f"demos_{n}_misleading", "safety", base_def, n_demos=n, misleading=True, demos=recs))

    # --- steerability: safety-policy edits -----------------------------------
    if ("steerability" in want) or ("steer_safety" in want):
        defs = list(D.STEERABILITY_DEFINITIONS[family])
        if family in _CATEGORY_FAMILIES and "none" not in defs:
            defs = ["none"] + defs  # policy-ablation lives alongside the variants
        for name in defs:
            conditions.append(Condition("steer_safety", f"def_{name}", "safety", name))

    # --- steerability: classification reframe --------------------------------
    if (("steerability" in want) or ("steer_classification" in want)) and _CLS_DEFS[family]:
        for name in _CLS_DEFS[family]:
            conditions.append(Condition("steer_classification", f"cls_{name}", "classification", name))

    return conditions


def context_values(eval_df: pd.DataFrame, cond: Condition, seed: int) -> list[str]:
    """Per-sample context strings for a condition (handles global shuffling)."""
    ctx = eval_df["context"].fillna("").astype(str).tolist()
    if cond.context_mode == "correct":
        return ctx
    if cond.context_mode == "shuffled":
        rng = np.random.default_rng(seed)
        return list(rng.permutation(np.array(ctx, dtype=object)))
    return [""] * len(eval_df)  # "none"
