"""Plotting helpers for the analysis notebook.

Bars are grouped **by model**: each judge gets a cluster, and within a cluster
there is one bar per condition (coloured by condition). Each experiment has a
performance view (accuracy or F1, toggled via ``metric=``, with standard-
deviation error bars across seeds) and a flip-rate view.

All functions take the summary DataFrame (results/<dataset>__<mode>/summary.csv).
Kept out of the notebook so the plotting logic is importable and testable
(matplotlib 'Agg' works headless).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Condition -> human-readable legend label.
LABELS = {
    "context_none": "no context",
    "context_correct": "correct context",
    "context_shuffled": "shuffled context",
    "demos_0": "no demonstrations",
    "demos_1_correct": "1 correct", "demos_2_correct": "2 correct", "demos_4_correct": "4 correct",
    "demos_1_misleading": "1 misleading", "demos_2_misleading": "2 misleading", "demos_4_misleading": "4 misleading",
    "def_none": "no definition", "def_base": "base", "def_strict": "strict", "def_swapped": "swapped",
    "cls_base": "base", "cls_strict": "strict", "cls_swapped": "swapped",
}

_METRIC_LABEL = {"accuracy": "accuracy", "f1": "F1"}


def _judge_label(j: str) -> str:
    """Drop the litellm provider prefix for readability (e.g. 'ollama/llama3.2:3b' -> 'llama3.2:3b')."""
    return j.split("/", 1)[1] if "/" in j else j


def _shades(cmap_name: str, n: int) -> list:
    cmap = plt.get_cmap(cmap_name)
    return [cmap(0.4 + 0.5 * (i / max(n - 1, 1))) for i in range(n)]


def _present(df, order):
    have = set(df["condition"])
    return [c for c in order if c in have]


def _model_grouped(ax, df, conditions, value_col, *, std_col=None, colors=None,
                   title="", ylabel="", expected=False):
    """Grouped bars: one cluster per judge, one bar per condition within it."""
    judges = sorted(df["judge"].unique())
    if not judges or not conditions:
        ax.set_visible(False)
        return
    x = np.arange(len(judges))
    w = 0.8 / len(conditions)
    for ci, cond in enumerate(conditions):
        vals, errs = [], []
        for j in judges:
            r = df[(df["judge"] == j) & (df["condition"] == cond)]
            vals.append(r[value_col].iloc[0] if len(r) else np.nan)
            errs.append(r[std_col].iloc[0] if (std_col and len(r)) else np.nan)
        off = (ci - (len(conditions) - 1) / 2) * w
        color = colors[ci] if colors else None
        ax.bar(x + off, vals, w, yerr=(errs if std_col else None), capsize=2,
               color=color, label=LABELS.get(cond, cond))
        if expected:  # ideal-steerability flip level (same across judges) per condition
            ev = df[df["condition"] == cond]["expected_flip_rate"]
            ev = ev.iloc[0] if len(ev) else np.nan
            if not np.isnan(ev):
                for xi in x:
                    ax.plot([xi + off - w / 2, xi + off + w / 2], [ev, ev], color="black", lw=1.3)
    if expected:
        ax.plot([], [], color="black", lw=1.3, label="expected (ideal)")
    ax.set_xticks(x)
    ax.set_xticklabels([_judge_label(j) for j in judges])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, ncol=2 if len(conditions) > 4 else 1)


# Condition orders + colour schemes per experiment.
_CTX = ["context_none", "context_correct", "context_shuffled"]
_CTX_COLORS = ["0.6", "#2ca02c", "#ff7f0e"]

_DEMOS = ["demos_0", "demos_1_correct", "demos_2_correct", "demos_4_correct",
          "demos_1_misleading", "demos_2_misleading", "demos_4_misleading"]
_DEMOS_COLORS = ["0.6", *_shades("Blues", 3), *_shades("Reds", 3)]

_STEER_COLORS = list(plt.get_cmap("tab10").colors)


def _ordered(df, order):
    present = _present(df, order)
    return present, [order.index(c) for c in present]


# --- susceptibility: context -------------------------------------------------
def plot_context_performance(summary: pd.DataFrame, metric: str = "accuracy"):
    df = summary[summary["experiment"] == "susc_context"]
    if df.empty:
        print("No context experiment (dataset has no context column).")
        return None
    conds = _present(df, _CTX)
    colors = [_CTX_COLORS[_CTX.index(c)] for c in conds]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    _model_grouped(ax, df, conds, f"{metric}_mean", std_col=f"{metric}_std", colors=colors,
                   title=f"Susceptibility to novel context — {_METRIC_LABEL[metric]}",
                   ylabel=_METRIC_LABEL[metric])
    fig.tight_layout()
    return fig


def plot_context_flips(summary: pd.DataFrame):
    df = summary[summary["experiment"] == "susc_context"]
    if df.empty:
        print("No context experiment (dataset has no context column).")
        return None
    # flip rate is relative to no-context, so only correct/shuffled carry a value
    conds = _present(df, ["context_correct", "context_shuffled"])
    colors = [_CTX_COLORS[_CTX.index(c)] for c in conds]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    _model_grouped(ax, df, conds, "flip_rate", colors=colors,
                   title="Susceptibility to novel context — flip rate", ylabel="flip rate")
    fig.tight_layout()
    return fig


# --- susceptibility: demonstrations ------------------------------------------
def plot_demos_performance(summary: pd.DataFrame, metric: str = "accuracy"):
    df = summary[summary["experiment"] == "susc_demos"]
    if df.empty:
        print("No demonstrations experiment.")
        return None
    conds = _present(df, _DEMOS)
    colors = [_DEMOS_COLORS[_DEMOS.index(c)] for c in conds]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    _model_grouped(ax, df, conds, f"{metric}_mean", std_col=f"{metric}_std", colors=colors,
                   title=f"Susceptibility to demonstrations — {_METRIC_LABEL[metric]}",
                   ylabel=_METRIC_LABEL[metric])
    fig.tight_layout()
    return fig


def plot_demos_flips(summary: pd.DataFrame):
    df = summary[summary["experiment"] == "susc_demos"]
    if df.empty:
        print("No demonstrations experiment.")
        return None
    conds = _present(df, [c for c in _DEMOS if c != "demos_0"])  # base has flip 0 by definition
    colors = [_DEMOS_COLORS[_DEMOS.index(c)] for c in conds]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    _model_grouped(ax, df, conds, "flip_rate", colors=colors,
                   title="Susceptibility to demonstrations — flip rate", ylabel="flip rate")
    fig.tight_layout()
    return fig


# --- steerability ------------------------------------------------------------
def _steer_frames(summary):
    return (summary[summary["experiment"] == "steer_safety"],
            summary[summary["experiment"] == "steer_classification"])


def _steer_fig(summary, value_col, *, std_col, ylabel, title_metric, expected):
    safety, cls = _steer_frames(summary)
    if safety.empty and cls.empty:
        print("No steerability experiment.")
        return None
    ncols = 2 if not cls.empty else 1
    fig, axes = plt.subplots(1, ncols, figsize=(8 * ncols, 4.5), squeeze=False)

    s_conds = list(dict.fromkeys(safety["condition"]))
    s_colors = _STEER_COLORS[: len(s_conds)]
    _model_grouped(axes[0][0], safety, s_conds, value_col, std_col=std_col, colors=s_colors,
                   title=f"Steerability: safety definitions — {title_metric}", ylabel=ylabel,
                   expected=expected)
    if ncols == 2:
        c_conds = list(dict.fromkeys(cls["condition"]))
        c_colors = _STEER_COLORS[: len(c_conds)]
        _model_grouped(axes[0][1], cls, c_conds, value_col, std_col=std_col, colors=c_colors,
                       title=f"Steerability: classification reframing — {title_metric}",
                       ylabel=ylabel, expected=expected)
    fig.tight_layout()
    return fig


def plot_steerability_performance(summary: pd.DataFrame, metric: str = "accuracy"):
    return _steer_fig(summary, f"{metric}_mean", std_col=f"{metric}_std",
                      ylabel=_METRIC_LABEL[metric], title_metric=_METRIC_LABEL[metric], expected=False)


def plot_steerability_flips(summary: pd.DataFrame):
    return _steer_fig(summary, "flip_rate", std_col=None,
                      ylabel="flip rate", title_metric="flip rate", expected=True)
