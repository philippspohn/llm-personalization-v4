"""Evaluation helpers for `run_benchmark_v2`.

We piggy-back on the v1 `PersonalizationAttributeJudge`, which already does
the right thing: for each test user, judge their generated response against
their list of GT (attribute, side) targets, with the `11 - score` flip for
`avoid` sides, and average per user.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from llm_personalization.benchmark.attribute_benchmark.attribute_personalization_judge import (
    PersonalizationAttributeJudge,
)
from llm_personalization.judge.judge import AttributeJudge

from .dataset import Message, PersonalizationExample


@dataclass
class EvaluationSummary:
    n_users: int
    score_mean: float
    score_std: float
    baseline_mean: float | None = None
    baseline_std: float | None = None
    win_rate: float | None = None
    tie_rate: float | None = None
    loss_rate: float | None = None

    def as_dict(self) -> dict:
        return {
            "n_users": self.n_users,
            "score_mean": self.score_mean,
            "score_std": self.score_std,
            "baseline_mean": self.baseline_mean,
            "baseline_std": self.baseline_std,
            "win_rate": self.win_rate,
            "tie_rate": self.tie_rate,
            "loss_rate": self.loss_rate,
        }


def _build_judge_inputs(
    test_examples: list[PersonalizationExample],
    responses: list[str],
) -> tuple[list[str], list[list[Message]]]:
    if len(test_examples) != len(responses):
        raise ValueError(
            f"len(test_examples)={len(test_examples)} != len(responses)={len(responses)}"
        )
    user_ids = [ex.user_id for ex in test_examples]
    convs = [
        ex.prompt + [{"role": "assistant", "content": resp}]
        for ex, resp in zip(test_examples, responses)
    ]
    return user_ids, convs


def judge_responses(
    *,
    test_examples: list[PersonalizationExample],
    responses: list[str],
    attribute_judge: AttributeJudge,
    user_id_to_gt: dict[str, list[dict[str, str]]] | None = None,
) -> list[float]:
    """Return one score per test user for the given generated responses.

    The `attribute_judge` MUST already be loaded by the caller (lifecycle
    is the orchestrator's responsibility, since model load/unload is the
    expensive step).
    """
    if user_id_to_gt is None:
        user_id_to_gt = {ex.user_id: ex.gt_user_attributes for ex in test_examples}

    pj = PersonalizationAttributeJudge(
        attribute_judge=attribute_judge,
        user_id_to_response_style_attributes=user_id_to_gt,
    )
    user_ids, convs = _build_judge_inputs(test_examples, responses)
    return pj.judge(user_ids, convs)


def summarize(
    scores: list[float],
    baseline_scores: list[float] | None = None,
) -> EvaluationSummary:
    import statistics

    if not scores:
        raise ValueError("Empty scores list")

    mean = float(statistics.fmean(scores))
    std = float(statistics.pstdev(scores)) if len(scores) > 1 else 0.0

    s = EvaluationSummary(n_users=len(scores), score_mean=mean, score_std=std)

    if baseline_scores is not None:
        if len(baseline_scores) != len(scores):
            raise ValueError(
                f"len(baseline)={len(baseline_scores)} != len(scores)={len(scores)}"
            )
        s.baseline_mean = float(statistics.fmean(baseline_scores))
        s.baseline_std = float(statistics.pstdev(baseline_scores)) if len(baseline_scores) > 1 else 0.0
        wins = sum(1 for a, b in zip(scores, baseline_scores) if a > b)
        ties = sum(1 for a, b in zip(scores, baseline_scores) if a == b)
        s.win_rate = wins / len(scores)
        s.tie_rate = ties / len(scores)
        s.loss_rate = 1.0 - s.win_rate - s.tie_rate
    return s
