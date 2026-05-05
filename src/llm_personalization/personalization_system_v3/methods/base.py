from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..cache import CachedDataset


class PersonalizationSystemV3(ABC):
    @abstractmethod
    def train(
        self,
        dataset: CachedDataset,
        user_id_to_weighted_attrs: dict[str, list[dict]],
        save_path: Path,
        val_dataset: CachedDataset | None = None,
        val_user_id_to_weighted_attrs: dict[str, list[dict]] | None = None,
    ) -> None:
        ...

    @abstractmethod
    def evaluate(
        self,
        dataset: CachedDataset,
        load_path: Path,
        user_id_to_weighted_attrs: dict[str, list[dict]] | None = None,
    ) -> list[str]:
        """Returns one response string per user (in dataset order).

        `user_id_to_weighted_attrs` is the test-split GT; required by oracle,
        ignored by trained methods.
        """
        ...
