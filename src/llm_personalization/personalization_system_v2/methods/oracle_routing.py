"""Oracle routing: at test time, generate using the user's GT (attribute, side).

This is the single-attribute oracle baseline. For each test user we pick
*one* GT (attribute, side) pair (the first one if there are several) and
inject it into the same Nemotron system-prompt template that was used to
generate the cached candidate responses. No training. Provides an upper
bound for the routing family.
"""

from __future__ import annotations

from llm_personalization.llm.llm_helper import LLMHelper

from ..dataset import PersonalizationExample
from ..method import MethodInput, PersonalizationMethod
from ..prompts import format_system_prompt


class OracleRouting(PersonalizationMethod):
    needs_gt_at_test = True

    def __init__(self, llm: LLMHelper):
        self.llm = llm

    def fit(self, train: list[PersonalizationExample]) -> None:
        return  # no training

    def generate(self, batch: list[MethodInput]) -> list[str]:
        prompts = []
        for item in batch:
            if not item.gt_user_attributes:
                raise ValueError(
                    f"OracleRouting requires gt_user_attributes; user_id={item.user_id} "
                    f"has none. Did the orchestrator forget to populate them?"
                )
            gt = item.gt_user_attributes[0]                       # use the first GT pair
            sys_prompt = format_system_prompt(gt["attribute"], gt["side"])
            prompts.append([{"role": "system", "content": sys_prompt}, *item.prompt])
        self.llm.load()
        try:
            responses = self.llm.generate(prompts)                # one batched vLLM call
        finally:
            self.llm.unload()
        return [r.content for r in responses]
