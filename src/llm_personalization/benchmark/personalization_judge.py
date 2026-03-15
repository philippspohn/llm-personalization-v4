from abc import ABC, abstractmethod
from typing import Any

class PersonalizationJudge(ABC):
    def __init__(self):
        self.reset_train_statistics()

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def unload(self):
        pass

    def get_train_statistics(self) -> dict[str, Any]:
        return self.train_statistics

    def reset_train_statistics(self):
        """
        Should not be called by personalization system.
        """
        self.train_statistics = {
            "num_judge_requests": 0,
            "requested_ids": [],
        }

    @abstractmethod
    def judge(self, user_ids: list[str], conversations: list[list[dict[str, str]]]) -> list[float]:
        pass