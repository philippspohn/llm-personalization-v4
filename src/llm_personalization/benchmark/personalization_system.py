from pathlib import Path
from datasets import Dataset
from typing import Protocol
from dataclasses import dataclass
from abc import ABC, abstractmethod
from llm_personalization.judge.judge import AttributeJudge
from llm_personalization.benchmark.personalization_judge import PersonalizationJudge

@dataclass
class PersonalizationItem:
    user_id: str
    conversation_history: list[list[dict[str, str]]]
    current_messages: list[dict[str, str]] # TODO: multiple test messages?

class PersonalizationDataset(Protocol):
    def __len__(self) -> int:
        pass

    def __getitem__(self, index: int) -> PersonalizationItem:
        pass

class PersonalizationSystem(ABC):
    @abstractmethod
    def train(self, dataset: PersonalizationDataset, judge: PersonalizationJudge, save_path: Path):
        """
        Trains the system on a given dataset and judge, saves the trained system to a given path (and unloads it).
        """
        pass

    @abstractmethod
    def evaluate(self, dataset: PersonalizationDataset, load_path: Path) -> list[str]:
        """
        Evaluates the system on a given dataset, and returns the scores.
        """
        pass