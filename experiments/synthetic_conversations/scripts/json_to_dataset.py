"""
Convert synthetic conversations JSONL files to a HuggingFace dataset.

Each JSONL record (one user) is converted to a dataset row with:
  - user_id: str(user_idx)
  - rewrite_style_attributes: list of {attribute, side} dicts
  - conversation_history: all but the last conversation (list of message lists)
  - current_messages: the last conversation's messages

Usage:
    python json_to_dataset.py \
        --train_jsonl data/synthetic_conversations/gpt-oss-120b_k10_1attrs_16000_1000_train.jsonl \
        --test_jsonl  data/synthetic_conversations/gpt-oss-120b_k10_1attrs_16000_1000_test.jsonl \
        --output_path data/synthetic_conversations/gpt-oss-120b_k10_1attrs_16000_1000_hf
"""

import json
import argparse
from pathlib import Path

from datasets import Dataset, DatasetDict


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def convert_records(records: list[dict]) -> dict[str, list]:
    user_ids = []
    user_attributes = []
    conversation_history = []
    current_messages = []

    for record in records:
        conversations = record["conversations"]
        user_ids.append(str(record["user_idx"]))
        user_attributes.append(record["rewrite_style_attributes"])
        conversation_history.append([conv["messages"] for conv in conversations[:-1]])
        current_messages.append(conversations[-1]["messages"])

    return {
        "user_id": user_ids,
        "user_attributes": user_attributes,
        "conversation_history": conversation_history,
        "current_messages": current_messages,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert synthetic conversations JSONL to HF dataset")
    parser.add_argument("--train_jsonl", type=str, required=True, help="Path to train JSONL file")
    parser.add_argument("--test_jsonl", type=str, required=True, help="Path to test JSONL file")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for HF dataset")
    args = parser.parse_args()

    train_path = Path(args.train_jsonl)
    test_path = Path(args.test_jsonl)
    output_path = Path(args.output_path)

    print(f"Loading train from {train_path}...")
    train_records = load_jsonl(train_path)
    print(f"  {len(train_records)} users")

    print(f"Loading test from {test_path}...")
    test_records = load_jsonl(test_path)
    print(f"  {len(test_records)} users")

    print("Converting train split...")
    train_dataset = Dataset.from_dict(convert_records(train_records))

    print("Converting test split...")
    test_dataset = Dataset.from_dict(convert_records(test_records))

    dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})

    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving dataset to {output_path}...")
    dataset_dict.save_to_disk(str(output_path))

    print("Done.")
    print(f"  train: {len(train_dataset)} rows")
    print(f"  test:  {len(test_dataset)} rows")
    print(f"  Features: {train_dataset.features}")


if __name__ == "__main__":
    main()
