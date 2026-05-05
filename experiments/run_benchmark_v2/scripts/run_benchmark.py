"""Top-level orchestrator for `run_benchmark_v2`.

Pipeline:
  1. Load the materialized BenchmarkDataset (must have been prepared by
     `prepare_dataset.py`).
  2. Instantiate the personalization method via Hydra and call `fit` on the
     train examples.
  3. Build batched MethodInputs from the test split (with GT injected only
     when the method declares `needs_gt_at_test`).
  4. `method.generate(batch)` -> one batched call (the method is responsible
     for batching internally).
  5. Optionally run the unpersonalized baseline as a reference (also one
     batched call).
  6. Load the judge once and score both methodperson + baseline (two batched
     judge passes).
  7. Save per-user results JSONL + summary metrics JSON.

All heavy work is one (or few) batched LLM/judge calls; there are no
per-user LLM round-trips on the hot path.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf

from llm_personalization.judge.judge import AttributeJudge
from llm_personalization.llm.llm_helper import LLMHelper
from llm_personalization.personalization_system_v2.dataset import BenchmarkDataset, PersonalizationExample
from llm_personalization.personalization_system_v2.evaluator import (
    judge_responses,
    summarize,
)
from llm_personalization.personalization_system_v2.method import MethodInput, PersonalizationMethod
from llm_personalization.personalization_system_v2.methods.baseline import UnpersonalizedBaseline


def _build_inputs(
    examples: list[PersonalizationExample],
    method: PersonalizationMethod,
) -> list[MethodInput]:
    """Build MethodInputs, populating GT only if the method opts in."""
    inputs: list[MethodInput] = []
    for ex in examples:
        gt = ex.gt_user_attributes if method.needs_gt_at_test else None
        inputs.append(
            MethodInput(
                user_id=ex.user_id,
                history=ex.history,
                prompt=ex.prompt,
                gt_user_attributes=gt,
            )
        )
    return inputs


def _resolve_output_dir(template: str, dataset_name: str, method_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(
        template
        .replace("{dataset_name}", dataset_name)
        .replace("{method_name}", method_name)
        .replace("{timestamp}", timestamp)
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    project_root = Path(get_original_cwd())
    materialized_dir = project_root / cfg.dataset.materialized_dir

    if cfg.require_materialized and not (materialized_dir / "meta.json").exists():
        raise FileNotFoundError(
            f"Materialized dataset not found at {materialized_dir}. "
            f"Run `python experiments/run_benchmark_v2/scripts/prepare_dataset.py "
            f"dataset={cfg.dataset.name.split('_')[1]}` first."
        )

    print(f"[benchmark_v2] Loading materialized dataset from {materialized_dir}", flush=True)
    bench = BenchmarkDataset.load(materialized_dir)
    print(
        f"[benchmark_v2]   {len(bench.train)} train / {len(bench.test)} test users; "
        f"{len(bench.attributes)} attributes x {len(bench.sides)} sides",
        flush=True,
    )

    test_examples = bench.test
    if cfg.get("test_limit") is not None:
        test_examples = test_examples[: int(cfg.test_limit)]
        print(f"[benchmark_v2]   capped to {len(test_examples)} test users", flush=True)

    method_name = cfg.method._target_.split(".")[-1]
    dataset_name = bench.name
    output_dir = project_root / _resolve_output_dir(cfg.output_dir, dataset_name, method_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[benchmark_v2] Writing results to {output_dir}", flush=True)

    # Save the resolved config for reproducibility.
    with open(output_dir / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    # ------------------------------------------------------------------ method
    method: PersonalizationMethod = instantiate(cfg.method)
    print(
        f"[benchmark_v2] Method = {method.__class__.__name__} "
        f"(needs_gt_at_test={method.needs_gt_at_test})",
        flush=True,
    )

    # Filter train examples for fitting: most methods need cached ratings,
    # and we cap to `fit_train_limit` so the same fit-set is used everywhere
    # (the rest is reserved for later validation).
    fit_examples = [ex for ex in bench.train if ex.has_ratings]
    if cfg.get("fit_train_limit") is not None:
        fit_examples = fit_examples[: int(cfg.fit_train_limit)]
    print(
        f"[benchmark_v2] Fitting method on {len(fit_examples)} rated train examples "
        f"(of {len(bench.train)} total; fit_train_limit={cfg.get('fit_train_limit')})...",
        flush=True,
    )
    method.fit(fit_examples)

    print(f"[benchmark_v2] Generating personalized responses for {len(test_examples)} test users...", flush=True)
    method_inputs = _build_inputs(test_examples, method)
    pers_responses = method.generate(method_inputs)
    if len(pers_responses) != len(test_examples):
        raise RuntimeError(
            f"method.generate returned {len(pers_responses)} responses, "
            f"expected {len(test_examples)}"
        )

    # ---------------------------------------------------------------- baseline
    baseline_responses: list[str] | None = None
    if cfg.get("run_baseline", True):
        print(f"[benchmark_v2] Generating unpersonalized baseline responses...", flush=True)
        baseline_llm: LLMHelper = instantiate(cfg.llm)
        baseline = UnpersonalizedBaseline(llm=baseline_llm)
        baseline.fit(bench.train)
        baseline_responses = baseline.generate(_build_inputs(test_examples, baseline))

    # ------------------------------------------------------------------- judge
    if cfg.get("run_judge", True):
        print(f"[benchmark_v2] Loading judge for scoring...", flush=True)
        judge: AttributeJudge = instantiate(cfg.judge)
        judge.load()
        try:
            print(f"[benchmark_v2]   Scoring personalized responses...", flush=True)
            pers_scores = judge_responses(
                test_examples=test_examples,
                responses=pers_responses,
                attribute_judge=judge,
            )
            baseline_scores: list[float] | None = None
            if baseline_responses is not None:
                print(f"[benchmark_v2]   Scoring baseline responses...", flush=True)
                baseline_scores = judge_responses(
                    test_examples=test_examples,
                    responses=baseline_responses,
                    attribute_judge=judge,
                )
        finally:
            judge.unload()

        # Replace any None scores with 0 for aggregation; flag count.
        n_none_pers = sum(1 for s in pers_scores if s is None)
        pers_scores_clean = [float(s) if s is not None else 0.0 for s in pers_scores]
        if baseline_scores is not None:
            n_none_base = sum(1 for s in baseline_scores if s is None)
            baseline_scores_clean = [float(s) if s is not None else 0.0 for s in baseline_scores]
        else:
            n_none_base = 0
            baseline_scores_clean = None

        summary = summarize(pers_scores_clean, baseline_scores_clean)
        summary_dict = summary.as_dict()
        summary_dict["n_none_personalized"] = n_none_pers
        summary_dict["n_none_baseline"] = n_none_base
    else:
        pers_scores_clean = [None] * len(test_examples)
        baseline_scores_clean = [None] * len(test_examples) if baseline_responses is not None else None
        summary_dict = {"n_users": len(test_examples), "judging": "skipped"}

    # ----------------------------------------------------------------- outputs
    per_user_records = []
    for i, ex in enumerate(test_examples):
        rec = {
            "user_id": ex.user_id,
            "gt_user_attributes": ex.gt_user_attributes,
            "prompt": ex.prompt,
            "personalized_response": pers_responses[i],
            "personalized_score": pers_scores_clean[i],
        }
        if baseline_responses is not None:
            rec["baseline_response"] = baseline_responses[i]
            rec["baseline_score"] = (
                baseline_scores_clean[i] if baseline_scores_clean is not None else None
            )
        per_user_records.append(rec)

    with open(output_dir / "results.jsonl", "w") as f:
        for rec in per_user_records:
            f.write(json.dumps(rec) + "\n")

    summary_dict.update(
        {
            "dataset": dataset_name,
            "method": method_name,
            "n_test_users": len(test_examples),
        }
    )
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary_dict, f, indent=2)

    print("[benchmark_v2] === Summary ===", flush=True)
    for k, v in summary_dict.items():
        print(f"  {k}: {v}", flush=True)
    print(f"[benchmark_v2] Done. Results in {output_dir}", flush=True)


if __name__ == "__main__":
    main()
