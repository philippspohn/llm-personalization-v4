"""Personalization-method ABC for `run_benchmark_v2`.

Methods are dumb consumers of cached candidate responses + ratings (see
`dataset.py`). The framework controls what's visible at fit and at generate
time; in particular, GT user attributes are only populated on test inputs
when the method explicitly opts in via `needs_gt_at_test`.

All methods MUST implement batched generation -- `generate(batch)` should
issue one (or a small number of) batched vLLM call(s), not one call per
user.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .dataset import Conversation, Message, PersonalizationExample


@dataclass
class MethodInput:
    """View of one test user passed to `PersonalizationMethod.generate`.

    `gt_user_attributes` is populated by the orchestrator iff the method
    declares `needs_gt_at_test = True`; otherwise it is `None` so methods
    cannot accidentally peek at the test labels.
    """

    user_id: str
    history: list[Conversation]
    prompt: list[Message]
    gt_user_attributes: list[dict[str, str]] | None = None


class PersonalizationMethod(ABC):
    """Base class for v2 personalization methods.

    Subclasses set `needs_gt_at_test` to `True` only if they are an oracle
    (or a controlled-leakage analysis variant). The orchestrator enforces
    this and refuses to populate GT otherwise.
    """

    needs_gt_at_test: bool = False

    @abstractmethod
    def fit(self, train: list[PersonalizationExample]) -> None:
        """Train / index from the cached training examples.

        May be a no-op (e.g. oracle, vanilla baseline). Heavy methods (DPO,
        RAG with embedding store) do their work here.
        """

    @abstractmethod
    def generate(self, batch: list[MethodInput]) -> list[str]:
        """Return one final response string per input, in input order.

        Implementations MUST batch into one (or few) vLLM calls; per-user
        generate calls are forbidden on the hot path.
        """
