"""Shared system-prompt template for attribute-conditioned generation.

Copied verbatim from
`llm_personalization.personalization_system.attribute_personalization
.attribute_personalization_system` so the v2 pipeline can be developed
independently of the v1 system. The cached candidate responses on disk
were produced with this exact template; routing / oracle methods must use
the same template at test time to stay in distribution.
"""

from __future__ import annotations


SYSTEM_PROMPT_TEMPLATE = """
Your task is to respond to the user's prompt while strictly adhering to the following response principle:
{direction} {attribute}

You must {direction_instruction}.
"""


def format_system_prompt(attribute: str, side: str) -> str:
    if side == "follow":
        return SYSTEM_PROMPT_TEMPLATE.format(
            direction="FOLLOW",
            attribute=attribute,
            direction_instruction=f"demonstrate strong {attribute} in your response",
        )
    if side == "avoid":
        return SYSTEM_PROMPT_TEMPLATE.format(
            direction="AVOID",
            attribute=attribute,
            direction_instruction=f"avoid any {attribute} in your response",
        )
    raise ValueError(f"side must be 'follow' or 'avoid', got {side!r}")
