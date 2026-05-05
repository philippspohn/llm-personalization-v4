"""Vanilla unpersonalized baseline.

`fit` is a no-op; `generate` returns a single batched call to the configured
LLM with no system prompt (i.e. prompt-only). Serves as the reference floor
for win/loss rates and as a smoke test for the orchestration code.
"""

from __future__ import annotations

from llm_personalization.llm.llm_helper import LLMHelper

from ..dataset import PersonalizationExample
from ..method import MethodInput, PersonalizationMethod


class UnpersonalizedBaseline(PersonalizationMethod):
    needs_gt_at_test = False

    def __init__(self, llm: LLMHelper):
        self.llm = llm

    def fit(self, train: list[PersonalizationExample]) -> None:
        return  # nothing to learn

    def generate(self, batch: list[MethodInput]) -> list[str]:
        prompts = [item.prompt for item in batch]                # 1 batched call
        self.llm.load()
        try:
            responses = self.llm.generate(prompts)
        finally:
            self.llm.unload()
        return [r.content for r in responses]
