from abc import ABC, abstractmethod

type Messages = list[dict[str, str]]


class PrincipleJudge(ABC):
    @abstractmethod
    def judge_principle(self, conversations: list[Messages], principles: list[str]) -> list[float]:
        """
        Judge the compliance of a list of the last message in a conversation with a given principle.
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        pass

    @abstractmethod
    def load(self) -> None:
        pass
