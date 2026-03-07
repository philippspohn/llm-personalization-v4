from abc import ABC, abstractmethod


type Messages = list[dict[str, str]]


class AttributeJudge(ABC):
    @abstractmethod
    def judge_response_attribute(self, conversations: list[Messages], attributes: list[str]) -> list[float | None]:
        """
        Judge the compliance of a list of the last message in a conversation with a given attribute.
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        pass

    @abstractmethod
    def load(self) -> None:
        pass
