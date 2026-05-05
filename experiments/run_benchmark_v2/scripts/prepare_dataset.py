"""One-shot join of the three upstream artifacts into a single per-user JSONL.

Reads:
  cfg.dataset.source.synthetic_conversations.{train,test}
  cfg.dataset.source.attribute_responses.{train,test}
  cfg.dataset.source.attribute_response_judging.{dir,glob}

Writes:
  cfg.dataset.materialized_dir/{train,test}.jsonl  (one PersonalizationExample
                                                    per line)
  cfg.dataset.materialized_dir/meta.json           (name, attributes, sides)

Idempotent: skips writes if the materialized files already exist unless
`force=True` is passed.
"""

from __future__ import annotations

from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from llm_personalization.personalization_system_v2.dataset import (
    BenchmarkDataset,
    join_one_split,
)


@hydra.main(version_base=None, config_path="../configs", config_name="prepare_dataset")
def main(cfg: DictConfig) -> None:
    project_root = Path(get_original_cwd())
    ds_cfg = cfg.dataset
    src = ds_cfg.source

    materialized_dir = project_root / ds_cfg.materialized_dir
    train_out = materialized_dir / "train.jsonl"
    test_out = materialized_dir / "test.jsonl"
    if train_out.exists() and test_out.exists() and not cfg.get("force", False):
        print(
            f"[prepare_dataset] {train_out} and {test_out} already exist. "
            f"Pass force=true to overwrite. Exiting.",
            flush=True,
        )
        return

    print(f"[prepare_dataset] Joining train split...", flush=True)
    train_examples = list(
        join_one_split(
            split="train",
            synth_path=project_root / src.synthetic_conversations.train,
            responses_path=project_root / src.attribute_responses.train,
            judging_dir=project_root / src.attribute_response_judging.dir,
            judging_glob=src.attribute_response_judging.glob,
            input_limit=int(ds_cfg.train_input_limit) if ds_cfg.get("train_input_limit") else None,
        )
    )
    print(f"[prepare_dataset]   {len(train_examples)} train users", flush=True)

    print(f"[prepare_dataset] Joining test split...", flush=True)
    test_examples = list(
        join_one_split(
            split="test",
            synth_path=project_root / src.synthetic_conversations.test,
            responses_path=project_root / src.attribute_responses.test,
            judging_dir=project_root / src.attribute_response_judging.dir,
            judging_glob=src.attribute_response_judging.glob,
            input_limit=None,                     # test is never capped
        )
    )
    print(f"[prepare_dataset]   {len(test_examples)} test users", flush=True)

    # Sanity: every example should have 2*K candidates and each candidate
    # should have 2*K ratings.
    expected_candidates = len(cfg.attributes) * len(ds_cfg.sides)
    n_with_full_candidates = sum(1 for e in train_examples if len(e.candidates) == expected_candidates)
    n_with_full_ratings = sum(
        1
        for e in train_examples
        if e.has_ratings and all(len(c.ratings) == expected_candidates for c in e.candidates)
    )
    print(
        f"[prepare_dataset]   train: {n_with_full_candidates}/{len(train_examples)} have all "
        f"{expected_candidates} candidates; {n_with_full_ratings}/{len(train_examples)} have full "
        f"{expected_candidates}x{expected_candidates} ratings",
        flush=True,
    )

    bench = BenchmarkDataset(
        name=ds_cfg.name,
        attributes=list(cfg.attributes),
        sides=list(ds_cfg.sides),
        train=train_examples,
        test=test_examples,
    )
    print(f"[prepare_dataset] Writing to {materialized_dir}", flush=True)
    bench.save(materialized_dir)
    print(f"[prepare_dataset] Done.", flush=True)


if __name__ == "__main__":
    main()
