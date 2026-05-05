"""Judge each unpersonalized response (from
`experiments/run_benchmark_v3_unpersonalized_responses`) against every
response attribute, using `ParsedRatingJudge`. Mirrors
`judge_attribute_responses.py` but the input has one response per user
(not 2K), so the output also has one row per user with a flat
`scores: [{attribute, side, score, derived}, ...]` of size 2K.

Slurm-array friendly: contiguous slicing across an input directory of
parts.
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from llm_personalization.judge.parsed_rating_judge import ParsedRatingJudge


def _iter_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_input(path: Path) -> list[dict]:
    """Load either a single .jsonl file or every .jsonl in a directory."""
    if path.is_file():
        return list(_iter_jsonl(path))
    out = []
    for child in sorted(path.glob("*.jsonl")):
        out.extend(_iter_jsonl(child))
    return out


def _slice_for_task(records: list, task_id: int, task_count: int) -> list:
    if task_count <= 0:
        raise ValueError(f"array_task_count must be >= 1, got {task_count}")
    if not (0 <= task_id < task_count):
        raise ValueError(f"array_task_id={task_id} out of range for task_count={task_count}")
    n = len(records)
    return records[(n * task_id) // task_count : (n * (task_id + 1)) // task_count]


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    project_root = Path(get_original_cwd())

    input_path = project_root / cfg.input_path
    output_dir = project_root / cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    attributes = list(cfg.attributes)
    task_id = int(cfg.array_task_id)
    task_count = int(cfg.array_task_count)

    print(f"[judge_unpers] Loading from {input_path}", flush=True)
    all_records = _load_input(input_path)
    print(f"[judge_unpers]   {len(all_records)} users total", flush=True)

    records = _slice_for_task(all_records, task_id, task_count)
    if cfg.get("limit") is not None:
        records = records[: int(cfg.limit)]

    out_path = output_dir / f"part{task_id:04d}_of{task_count:04d}.jsonl"
    print(f"[judge_unpers] Task {task_id}/{task_count}: {len(records)} users -> {out_path}", flush=True)

    # One judge call per (user, attribute_name); avoid side derived as 11-x.
    judge_conversations: list[list[dict[str, str]]] = []
    judge_attributes: list[str] = []
    for rec in records:
        conv = list(rec["current_messages"]) + [{"role": "assistant", "content": rec["response"]}]
        for attr in attributes:
            judge_conversations.append(conv)
            judge_attributes.append(attr)

    total_calls = len(judge_conversations)
    print(f"[judge_unpers] Total judge calls: {total_calls} (users * attributes)", flush=True)

    judge: ParsedRatingJudge = instantiate(cfg.judge)
    judge.load()
    try:
        scores = judge.judge_response_attribute(judge_conversations, judge_attributes)
    finally:
        judge.unload()

    n_attrs = len(attributes)
    none_count = sum(1 for s in scores if s is None)
    if none_count:
        print(f"[judge_unpers] WARNING: {none_count}/{total_calls} judge calls returned None", flush=True)

    print(f"[judge_unpers] Writing to {out_path}", flush=True)
    with open(out_path, "w") as f:
        for u_i, rec in enumerate(tqdm(records, desc="writing")):
            base = u_i * n_attrs
            materialized = []
            for a_i, attr in enumerate(attributes):
                follow = scores[base + a_i]
                avoid = None if follow is None else 11.0 - float(follow)
                materialized.append({"attribute": attr, "side": "follow", "score": follow, "derived": False})
                materialized.append({"attribute": attr, "side": "avoid", "score": avoid, "derived": True})
            f.write(json.dumps({
                "user_id": rec["user_id"],
                "current_messages": rec["current_messages"],
                "response": rec["response"],
                "scores": materialized,
            }) + "\n")

    print(f"[judge_unpers] Done. {len(records)} users, {total_calls} calls, {none_count} unparsed.", flush=True)


if __name__ == "__main__":
    main()
