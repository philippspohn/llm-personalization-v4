from llm_personalization.benchmark.personalization_system import PersonalizationItem, PersonalizationDataset
from pathlib import Path
from dataclasses import dataclass
import json


@dataclass
class PersonaPersonalizationLabeledItem:
    user_id: str
    conversation_history: list[list[dict[str, str]]]
    current_messages: list[dict[str, str]]
    persona: dict[str, str]
    formatted_persona: str


class PersonaPersonalizationLabeledDataset:
    def __init__(self, train_path: str, test_path: str, split: str = "train",
                 train_limit: int | None = None, test_limit: int | None = None,
                 history_max_len: int | None = None):
        path = train_path if split == "train" else test_path
        self.data = []
        with open(path) as f:
            for line in f:
                self.data.append(json.loads(line))
        limit = train_limit if split == "train" else test_limit
        if limit is not None:
            self.data = self.data[:limit]
        self.history_max_len = history_max_len

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index: int) -> PersonaPersonalizationLabeledItem:
        row = self.data[index]
        conversations = row["conversations"]
        # Last conversation is the test item, rest is history
        history = [conv["messages"] for conv in conversations[:-1]]
        if self.history_max_len is not None:
            history = history[:self.history_max_len]
        current_messages = conversations[-1]["messages"]
        # Keep only first user message as current prompt
        current_messages = [msg for msg in current_messages if msg["role"] == "user"][:1]
        # Keep only first user message per history conversation for cleaner classifier input
        history = [[msg for msg in conv if msg["role"] == "user"][:1] for conv in history]
        return PersonaPersonalizationLabeledItem(
            user_id=row["persona_uuid"],
            conversation_history=history,
            current_messages=current_messages,
            persona=row["persona"],
            formatted_persona=row["formatted_persona"],
        )


class PersonaPersonalizationDataset(PersonalizationDataset):
    def __init__(self, labeled_dataset: PersonaPersonalizationLabeledDataset):
        self.labeled_dataset = labeled_dataset

    def __len__(self) -> int:
        return len(self.labeled_dataset)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index: int) -> PersonalizationItem:
        row = self.labeled_dataset[index]
        return PersonalizationItem(
            user_id=row.user_id,
            conversation_history=row.conversation_history,
            current_messages=row.current_messages,
        )
