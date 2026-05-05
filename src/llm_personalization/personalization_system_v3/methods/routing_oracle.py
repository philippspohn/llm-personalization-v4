"""Routing upper bound: at test time, pick per user the (attr, side) that
maximizes the weighted user-simulator score on cached candidate responses.
No training, no classifier — just cache lookups + one batched generate."""
from __future__ import annotations

import math
from pathlib import Path

from llm_personalization.llm.llm_helper import LLMHelper
from llm_personalization.utils.gpu_monitor import log_gpu_usage

from ..cache import CachedDataset
from ..prompts import format_system_prompt
from .base import PersonalizationSystemV3
from .routing import _weighted_user_score


class RoutingOracleSystem(PersonalizationSystemV3):
    def __init__(
        self,
        llm_helper_config: dict,
        attributes: list[str],
    ):
        self.attributes = list(attributes)
        self.llm_helper = LLMHelper(**llm_helper_config)

    def train(
        self,
        dataset: CachedDataset,
        user_id_to_weighted_attrs: dict[str, list[dict]],
        save_path: Path,
        val_dataset: CachedDataset | None = None,
        val_user_id_to_weighted_attrs: dict[str, list[dict]] | None = None,
    ) -> None:
        # No-op: oracle uses test-time GT directly. val ignored.
        Path(save_path).mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        dataset: CachedDataset,
        load_path: Path,
        user_id_to_weighted_attrs: dict[str, list[dict]] | None = None,
    ) -> list[str]:
        if user_id_to_weighted_attrs is None:
            raise ValueError("RoutingOracleSystem.evaluate requires user_id_to_weighted_attrs")

        print("[RoutingOracleSystem] 1. Picking oracle (attr, side) per user from cache...")
        prompts = []
        fallback = 0
        for user in dataset:
            targets = user_id_to_weighted_attrs.get(user.user_id, [])
            best_attr, best_side, best_score = None, None, -math.inf
            for attr in self.attributes:
                for side in ("follow", "avoid"):
                    s = _weighted_user_score(user, attr, side, targets)
                    if s is None:
                        continue
                    if s > best_score:
                        best_attr, best_side, best_score = attr, side, s
            if best_attr is None:
                fallback += 1
                best_attr, best_side = self.attributes[0], "follow"
            prompts.append(
                [{"role": "system", "content": format_system_prompt(best_attr, best_side)}]
                + user.current_messages
            )
        if fallback:
            print(f"[RoutingOracleSystem] {fallback} users had no cached scores; used fallback attribute")

        print("[RoutingOracleSystem] 2. Generating responses (single batched vLLM call)...")
        log_gpu_usage("Before LLM load")
        self.llm_helper.load()
        responses = self.llm_helper.generate(prompts)
        self.llm_helper.unload()
        log_gpu_usage("After LLM unload")
        return [r.content for r in responses]
