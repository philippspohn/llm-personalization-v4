from llm_personalization.benchmark.personalization_system import PersonalizationItem, PersonalizationDataset
from pathlib import Path
from datasets import load_from_disk
from dataclasses import dataclass


@dataclass
class PersonalizationLabeledItem:
    user_id: str
    conversation_history: list[list[dict[str, str]]]
    current_messages: list[dict[str, str]] # TODO: multiple test messages?
    user_attributes: list[dict[str, str]] # May not be used by the personalizaiton system # TODO: make nicer
    # example: [{"attribute": "enthusiastic", "side": "follow"}]


class AttributePersonalizationLabeledDataset(): # TODO: user pytorch dataset?
    def __init__(self, dataset_path: str, split: str = "train", train_limit: int | None = None, test_limit: int | None = None, history_max_len: int | None = None):
        self.dataset_path = Path(dataset_path)
        dataset_dict = load_from_disk(str(self.dataset_path))
        self.dataset = dataset_dict[split]
        self.history_max_len = history_max_len
        if split == "train" and train_limit is not None:
            self.dataset = self.dataset.select(range(train_limit))
        elif split == "test" and test_limit is not None:
            self.dataset = self.dataset.select(range(test_limit))

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index: int) -> PersonalizationLabeledItem:
        row = self.dataset[index]
        conversation_history = row["conversation_history"] if self.history_max_len is None else row["conversation_history"][:self.history_max_len]
        # Keep only the first user message per conversation for cleaner classifier input
        conversation_history = [[msg for msg in conv if msg["role"] == "user"][:1] for conv in conversation_history]
        return PersonalizationLabeledItem(
            user_id=row["user_id"],
            conversation_history=conversation_history,
            current_messages=row["current_messages"][:1],
            user_attributes=row["user_attributes"],
        )

class AttributePersonalizationDataset(PersonalizationDataset):
    def __init__(self, labeled_dataset: AttributePersonalizationLabeledDataset):
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