#!/usr/bin/env python3
"""Run susceptibility & steerability experiments for one or more judges.

Examples
--------
Offline smoke test (no API key needed):
    python run_experiments.py --dataset novelprompts --mode prompt --judges mock --seeds 2

Real judges (keys read from env by litellm, e.g. OPENAI_API_KEY):
    python run_experiments.py --dataset novelprompts --mode prompt \
        --judges "openai/gpt-5,anthropic/claude-4-5-sonnet" --seeds 5

Custom data (canonical CSV, see preprocess.py):
    python run_experiments.py --dataset custom --data my.csv --mode completion --judges mock

Outputs: results/<dataset>__<mode>/predictions.csv  (+ summary.csv via metrics).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from jss import datasets, definitions, experiments, extract, metrics, templates  # noqa: E402
from jss.judges import make_judge  # noqa: E402

DEFAULT_MODE = {"novelprompts": "prompt", "sorrybench": "completion", "sports": "completion", "custom": "completion"}


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=["novelprompts", "sorrybench", "sports", "custom"])
    p.add_argument("--mode", choices=["prompt", "completion"], default=None, help="judge the prompt or the completion")
    p.add_argument("--judges", default="mock", help="comma-separated litellm model strings, or 'mock'")
    p.add_argument("--experiments", default="susceptibility,steerability",
                   help="comma list: susceptibility, steerability, or fine-grained names")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--out", default="results")
    p.add_argument("--data", default=None, help="CSV path for --dataset custom")
    p.add_argument("--policy", default=None,
                   help="YAML/JSON custom safety policy (custom categories) for --dataset custom")
    p.add_argument("--api-base", default=None,
                   help="base URL for a local OpenAI-compatible endpoint (vLLM, etc.)")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--sleep", type=float, default=0.0,
                   help="seconds to wait between calls to rate-limited API judges "
                        "(ignored for mock / local ollama judges). Use for free tiers.")
    p.add_argument("--max-samples", type=int, default=None, help="cap eval rows (debug)")
    p.add_argument("--holdout-seed", type=int, default=0, help="seed to generate the demonstrations hold-out split")
    args = p.parse_args()

    mode = args.mode or DEFAULT_MODE[args.dataset]
    judges = [j.strip() for j in args.judges.split(",") if j.strip()]
    exps = [e.strip() for e in args.experiments.split(",") if e.strip()]

    pol = None
    if args.policy:
        from jss import policy as policy_mod
        pol = policy_mod.load_policy(args.policy)
        print(f"Loaded custom policy '{pol.name}' with categories: {list(pol.categories)}")

    print(f"Loading dataset={args.dataset} mode={mode} ...")
    eval_df, demo_df = datasets.load(args.dataset, mode, seed=args.holdout_seed, data_path=args.data, policy=pol)
    family = eval_df.attrs["family"]
    if args.max_samples:
        eval_df = eval_df.head(args.max_samples).reset_index(drop=True)
        eval_df.attrs["family"] = family
    print(f"  family={family}  eval_rows={len(eval_df)}  demo_rows={len(demo_df)}")

    conds = experiments.build_conditions(eval_df, demo_df, family, mode, exps)
    print(f"  conditions ({len(conds)}): {[c.condition for c in conds]}")
    n_calls = len(judges) * len(conds) * args.seeds * len(eval_df)
    print(f"  judges={judges} seeds={args.seeds}  -> up to {n_calls} judge calls (deduped by identical prompts)")

    records = eval_df.to_dict("records")
    rows = []
    for model in judges:
        print(f"\n=== judge: {model} ===")
        judge_kw = {"temperature": args.temperature, "max_tokens": args.max_tokens}
        if args.api_base:
            judge_kw["api_base"] = args.api_base
        judge = make_judge(model, **judge_kw)
        # Throttle only rate-limited hosted judges (not mock / local ollama).
        throttle = args.sleep if (args.sleep > 0 and not model.startswith(("mock", "ollama"))) else 0.0
        cache: dict[tuple, str] = {}
        for cond in conds:
            for seed in range(args.seeds):
                ctx_vals = experiments.context_values(eval_df, cond, seed)
                for i, row in enumerate(records):
                    prompt_text = templates.render(
                        family=family, mode=mode, task=cond.task, definition=cond.definition,
                        prompt=row["prompt"], completion=row.get("completion", ""),
                        context=ctx_vals[i], with_context=cond.with_context,
                        demos=cond.demos or None, misleading=cond.misleading,
                    )
                    key = (model, cond.task, seed, _hash(prompt_text))
                    raw = cache.get(key)
                    if raw is None:
                        raw = judge.generate(prompt_text, task=cond.task, seed=seed)
                        cache[key] = raw
                        if throttle:
                            time.sleep(throttle)
                    pred = extract.extract(raw, cond.task)
                    gold = definitions.gold(row, family, cond.definition)
                    rows.append({
                        "dataset": args.dataset, "mode": mode, "family": family, "judge": model,
                        "experiment": cond.experiment, "condition": cond.condition,
                        "definition": cond.definition, "task": cond.task,
                        "n_demos": cond.n_demos, "misleading": cond.misleading,
                        "with_context": cond.with_context, "seed": seed,
                        "prompt_id": row["prompt_id"],
                        "prediction": "" if pred is None else int(pred),
                        "gold": int(gold), "raw_output": raw,
                    })
            print(f"  done {cond.condition}")

    out_dir = Path(args.out) / f"{args.dataset}__{mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df = pd.DataFrame(rows)
    pred_path = out_dir / "predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"\nWrote {len(pred_df)} prediction rows -> {pred_path}")

    summary = metrics.summarize(pred_df, family=family)
    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote summary ({len(summary)} rows) -> {summary_path}")
    print("\nDone. Open notebooks/analysis.ipynb to plot, or inspect summary.csv.")


if __name__ == "__main__":
    main()
