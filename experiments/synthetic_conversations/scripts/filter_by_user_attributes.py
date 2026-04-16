"""Filter a HuggingFace dataset to a subset of user attributes.

This is useful for reusing the existing k10 synthetic-conversation dataset for
smaller ablations such as "first 2 attributes" or "first 5 attributes" without
regenerating conversations.

Example:
    python experiments/synthetic_conversations/scripts/filter_by_user_attributes.py \
        --input_path data/synthetic_conversations/gpt-oss-120b_k10_1attrs_16000_1000_hf \
        --attributes_file experiments/synthetic_conversations/configs/attributes/user_prompt_attributes_k2.yaml \
        --output_path data/synthetic_conversations/gpt-oss-120b_k10_1attrs_16000_1000_first2_hf
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from datasets import DatasetDict, load_from_disk
from omegaconf import OmegaConf


def _load_allowed_attributes(path: Path) -> list[str]:
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if isinstance(config, list):
        return [str(item) for item in config]
    if not isinstance(config, dict):
        raise ValueError(f"Unsupported attribute config format in {path}")
    for key in ("attributes", "candidate_attributes"):
        if key in config:
            return [str(item) for item in config[key]]
    raise ValueError(
        f"Expected one of ['attributes', 'candidate_attributes'] in {path}, got keys: {list(config)}"
    )


def _row_is_allowed(row: dict, allowed_attributes: set[str]) -> bool:
    return all(attribute["attribute"] in allowed_attributes for attribute in row["user_attributes"])


def _attribute_counts(dataset_split) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in dataset_split:
        for attribute in row["user_attributes"]:
            counts[attribute["attribute"]] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_path", type=Path, required=True, help="Input HF dataset path")
    parser.add_argument("--output_path", type=Path, required=True, help="Output HF dataset path")
    parser.add_argument(
        "--attributes_file",
        type=Path,
        required=True,
        help="YAML file containing the allowed attribute list",
    )
    args = parser.parse_args()

    allowed_attributes = _load_allowed_attributes(args.attributes_file)
    allowed_attribute_set = set(allowed_attributes)

    dataset_dict = load_from_disk(str(args.input_path))
    if not isinstance(dataset_dict, DatasetDict):
        raise ValueError(f"Expected a DatasetDict at {args.input_path}, got {type(dataset_dict)}")

    print(f"Loaded dataset from {args.input_path}")
    print(f"Keeping attributes: {allowed_attributes}")

    filtered_splits = {}
    for split_name, split_dataset in dataset_dict.items():
        filtered_split = split_dataset.filter(
            lambda row: _row_is_allowed(row, allowed_attribute_set),
            desc=f"Filtering {split_name}",
        )
        filtered_splits[split_name] = filtered_split

        kept = len(filtered_split)
        total = len(split_dataset)
        counts = _attribute_counts(filtered_split)
        print(f"[{split_name}] kept {kept}/{total} rows")
        print(f"[{split_name}] attribute counts: {dict(counts)}")

    filtered_dataset = DatasetDict(filtered_splits)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_dataset.save_to_disk(str(args.output_path))
    print(f"Saved filtered dataset to {args.output_path}")


if __name__ == "__main__":
    main()
