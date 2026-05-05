"""Robustness evaluation for v3 personalization systems.

Mirrors the v1 robustness benchmark: we wrap a CachedDataset + a list of
RobustnessQuestions so that each item exposes a CachedUser-shaped object
with `current_messages` set to the MC-formatted question. Each v3 system's
`evaluate(...)` then runs unmodified, generating an answer letter conditioned
on the user's history.

We compare per-source accuracy against an unpersonalized baseline (just the
question, no user history) generated live with vLLM.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import hydra
from omegaconf import DictConfig

from llm_personalization.benchmark.robustness_benchmark.robustness_dataset import (
    RobustnessQuestion,
    format_mc_prompt,
    load_robustness_questions,
    parse_answer_letter,
)

from .cache import CachedDataset, CachedUser


@dataclass
class _RobustnessUser:
    """Duck-typed CachedUser for robustness items: same field names so v3
    methods that touch `user.history`, `.current_messages`, `.user_id`,
    `.responses`, `.ratings`, `.gt_user_attributes` work unchanged."""
    user_id: str
    history: list[str]
    current_messages: list[dict]
    responses: dict
    ratings: dict
    gt_user_attributes: list[dict]


class V3RobustnessDataset:
    """Wraps a CachedDataset + list[RobustnessQuestion]. One item per
    question; each question is paired with a different user (cycling)."""

    def __init__(self, base: CachedDataset, questions: list[RobustnessQuestion]):
        self.base = base
        self.questions = questions
        self._users: list[_RobustnessUser] = []
        for i, q in enumerate(questions):
            base_user = base[i % len(base)]
            self._users.append(_RobustnessUser(
                user_id=base_user.user_id,
                history=base_user.history,
                current_messages=[{"role": "user", "content": format_mc_prompt(q)}],
                responses=base_user.responses,
                ratings=base_user.ratings,
                gt_user_attributes=base_user.gt_user_attributes,
            ))

    def __len__(self) -> int:
        return len(self._users)

    def __getitem__(self, i: int) -> _RobustnessUser:
        return self._users[i]

    def __iter__(self):
        return iter(self._users)

    def get_question(self, i: int) -> RobustnessQuestion:
        return self.questions[i]


def _grade(responses: list[str], questions: list[RobustnessQuestion]) -> tuple[dict, dict]:
    """Returns (correct, total) where mmlu_pro/truthfulqa entries are ints
    and bbq is a nested dict with `__total__` plus one entry per category."""
    correct: dict = {"mmlu_pro": 0, "truthfulqa": 0, "bbq": {"__total__": 0}}
    total: dict = {"mmlu_pro": 0, "truthfulqa": 0, "bbq": {"__total__": 0}}
    for resp, q in zip(responses, questions):
        is_correct = parse_answer_letter(resp) == q.correct_letter
        if q.source == "bbq":
            cat = (q.metadata or {}).get("category", "unknown")
            total["bbq"]["__total__"] += 1
            total["bbq"][cat] = total["bbq"].get(cat, 0) + 1
            if is_correct:
                correct["bbq"]["__total__"] += 1
                correct["bbq"][cat] = correct["bbq"].get(cat, 0) + 1
        else:
            total[q.source] += 1
            if is_correct:
                correct[q.source] += 1
    return correct, total


def unpers_cache_key(robustness_cfg) -> str:
    """Deterministic filename for the unpers MC cache, derived from which
    sources + limits the robustness eval is using."""
    parts = []
    for src in ("mmlu_pro", "truthfulqa", "bbq"):
        if robustness_cfg.get(f"include_{src}", False):
            limit = robustness_cfg.get(f"{src}_limit", None)
            parts.append(f"{src}{limit if limit is not None else 'full'}")
    parts.append(f"seed{robustness_cfg.get('seed', 42)}")
    return "_".join(parts) if parts else "empty"


def load_unpers_responses(path: Path, questions: list[RobustnessQuestion]) -> list[str]:
    """Load cached unpers MC responses, ordered to match `questions`. Errors
    if the cache is missing or doesn't cover all question_ids."""
    if not path.exists():
        raise FileNotFoundError(
            f"Unpersonalized robustness cache not found: {path}\n"
            f"Build it once per gen_model with experiments/run_benchmark_v3_unpersonalized_robustness."
        )
    by_id: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                by_id[rec["question_id"]] = rec["response"]
    missing = [q.question_id for q in questions if q.question_id not in by_id]
    if missing:
        raise ValueError(
            f"Unpers cache at {path} is missing {len(missing)} question_ids "
            f"(first: {missing[:3]}). Re-run the prep job with the same robustness config."
        )
    return [by_id[q.question_id] for q in questions]


def _summarize(correct: dict, total: dict, prefix: str = "") -> dict:
    out: dict = {}
    for src in ("mmlu_pro", "truthfulqa"):
        if total[src] > 0:
            acc = correct[src] / total[src]
            out[src] = {"correct": correct[src], "total": total[src], "accuracy": acc}
            print(f"      {prefix}{src}: {correct[src]}/{total[src]} = {acc:.4f}")
    if total["bbq"]["__total__"] > 0:
        bbq_out: dict = {}
        agg_c, agg_t = correct["bbq"]["__total__"], total["bbq"]["__total__"]
        bbq_out["aggregate"] = {"correct": agg_c, "total": agg_t, "accuracy": agg_c / agg_t}
        print(f"      {prefix}bbq (aggregate): {agg_c}/{agg_t} = {agg_c/agg_t:.4f}")
        for cat in sorted(k for k in total["bbq"] if k != "__total__"):
            c, t = correct["bbq"].get(cat, 0), total["bbq"][cat]
            bbq_out[cat] = {"correct": c, "total": t, "accuracy": c / t}
            print(f"        {prefix}bbq/{cat}: {c}/{t} = {c/t:.4f}")
        out["bbq"] = bbq_out
    return out


def run_robustness(
    system,
    test_dataset: CachedDataset,
    world_path: Path,
    world_idx,
    robustness_cfg: DictConfig,
    unpers_cache_path: Path,
    output_dir: Path,
    save_test_results: bool,
    user_id_to_weighted_attrs: dict[str, list[dict]] | None = None,
) -> dict:
    """Run the robustness benchmark for a single (system, world) pair.
    `unpers_cache_path` must point at a jsonl file produced by
    `experiments/run_benchmark_v3_unpersonalized_robustness` for the same
    (gen_model, robustness_cfg) — never regenerated here.
    Returns {"unpersonalized": ..., "personalized": ...}.
    """
    print(f"[BenchmarkV3-Robustness] world {world_idx}: loading questions...")
    questions = load_robustness_questions(
        include_mmlu_pro=robustness_cfg.get("include_mmlu_pro", True),
        include_truthfulqa=robustness_cfg.get("include_truthfulqa", True),
        include_bbq=robustness_cfg.get("include_bbq", False),
        mmlu_pro_limit=robustness_cfg.get("mmlu_pro_limit", None),
        truthfulqa_limit=robustness_cfg.get("truthfulqa_limit", None),
        bbq_limit=robustness_cfg.get("bbq_limit", None),
        seed=robustness_cfg.get("seed", 42),
    )
    print(f"[BenchmarkV3-Robustness] loaded {len(questions)} questions")

    # ---- unpersonalized baseline: pure cache load ----
    print(f"[BenchmarkV3-Robustness] loading unpers MC responses from {unpers_cache_path}")
    unpers_responses = load_unpers_responses(unpers_cache_path, questions)
    unpers_correct, unpers_total = _grade(unpers_responses, questions)
    print("[BenchmarkV3-Robustness] Unpersonalized:")
    unpers_summary = _summarize(unpers_correct, unpers_total, prefix="  ")

    # ---- personalized system ----
    print(f"[BenchmarkV3-Robustness] running {robustness_cfg.get('system_name', 'system')}.evaluate on robustness dataset...")
    robustness_dataset = V3RobustnessDataset(test_dataset, questions)
    pers_responses = system.evaluate(
        robustness_dataset, world_path, user_id_to_weighted_attrs=user_id_to_weighted_attrs,
    )
    pers_correct, pers_total = _grade(pers_responses, questions)

    print("[BenchmarkV3-Robustness] Personalized:")
    pers_summary = _summarize(pers_correct, pers_total, prefix="  ")
    for src in pers_summary:
        if src == "bbq":
            for sub in pers_summary[src]:
                base_acc = unpers_summary.get(src, {}).get(sub, {}).get("accuracy", 0.0)
                delta = pers_summary[src][sub]["accuracy"] - base_acc
                pers_summary[src][sub]["delta_vs_unpersonalized"] = delta
                print(f"      delta bbq/{sub}: {delta:+.4f}")
        else:
            base_acc = unpers_summary.get(src, {}).get("accuracy", 0.0)
            delta = pers_summary[src]["accuracy"] - base_acc
            pers_summary[src]["delta_vs_unpersonalized"] = delta
            print(f"      delta {src}: {delta:+.4f}")

    if save_test_results:
        records = []
        for i, (q, resp, base_resp) in enumerate(zip(questions, pers_responses, unpers_responses)):
            item = robustness_dataset[i]
            records.append({
                "world_idx": world_idx,
                "user_id": item.user_id,
                "history": item.history,
                "question_id": q.question_id,
                "source": q.source,
                "question_text": q.question_text,
                "options": q.options,
                "option_letters": q.option_letters,
                "correct_letter": q.correct_letter,
                "personalized_response": resp,
                "personalized_letter": parse_answer_letter(resp),
                "unpersonalized_response": base_resp,
                "unpersonalized_letter": parse_answer_letter(base_resp),
            })
        out_path = output_dir / f"robustness_world_{world_idx}_test_samples.jsonl"
        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"[BenchmarkV3-Robustness] saved {len(records)} records to {out_path}")

    return {"unpersonalized": unpers_summary, "personalized": pers_summary}
