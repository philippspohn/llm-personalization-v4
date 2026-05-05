"""On-disk benchmark dataset for `run_benchmark_v2`.

Each per-user record carries the full state every personalization method
needs: chat history, held-out prompt, the cached candidate responses, and
the cached judge ratings of every candidate against every (attribute, side)
target.

The dataset is materialized once by `prepare_dataset.py` -- a join of three
upstream artifacts -- and read directly from disk afterwards.

Source artifacts (all per-user, joined on user_id <-> user_idx):
  1. `data/synthetic_conversations/<name>_{train,test}.jsonl`
       -> chat history (`conversations[:-1]`) and the prompt
          (`conversations[-1]['messages'][:1]`).
  2. `data/attribute_responses/<name>_<gen_model>/{train,test}.jsonl`
       -> 2*K candidate responses per user (one per (attribute, side)).
  3. `data/attribute_response_judging/<name>_<gen_model>_<judge>/ratings_part*.jsonl`
       -> for each candidate, judge scores on all (attribute, side) targets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Literal


Message = dict[str, str]                  # {"role": "user"|"assistant"|"system", "content": str}
Conversation = list[Message]
Side = Literal["follow", "avoid"]


@dataclass
class Rating:
    target_attribute: str
    target_side: Side
    score: float                          # 1..10 scale (matches judge raw output / 10)
    derived: bool                         # True for synthesized opposite-side scores (11 - score)


@dataclass
class CandidateResponse:
    attribute: str                        # the (attribute, side) the candidate was generated under
    side: Side
    response: str
    finish_reason_stop: bool
    ratings: list[Rating]                 # 2*K entries, indexed by (target_attribute, target_side)

    def rating_dict(self) -> dict[tuple[str, Side], Rating]:
        return {(r.target_attribute, r.target_side): r for r in self.ratings}


@dataclass
class PersonalizationExample:
    user_id: str
    split: Literal["train", "test"]
    gt_user_attributes: list[dict[str, str]]   # [{"attribute": ..., "side": ...}, ...]
    history: list[Conversation]                # past conversations (excludes the held-out one)
    prompt: list[Message]                      # the held-out prompt (length 1: a single user turn)
    candidates: list[CandidateResponse]        # may be empty if no responses cached for this user

    @property
    def has_candidates(self) -> bool:
        return len(self.candidates) > 0

    @property
    def has_ratings(self) -> bool:
        return self.has_candidates and len(self.candidates[0].ratings) > 0


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def save_jsonl(path: Path, records: Iterable[dict]) -> int:
    n = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
            n += 1
    return n


# ---------------------------------------------------------------------------
# Source-record extraction (mirrors generate_attribute_responses.py)
# ---------------------------------------------------------------------------


def extract_history_and_prompt(synth_record: dict) -> tuple[list[Conversation], list[Message]]:
    """Split a `synthetic_conversations` record into (history, prompt).

    Mirrors `_extract_prompt_messages` in `generate_attribute_responses.py`:
    the prompt is the first user turn of the last conversation. Everything
    before that turn -- i.e. all earlier conversations in full -- becomes
    the history.
    """
    conversations = synth_record["conversations"]
    if not conversations:
        raise ValueError(f"user_idx={synth_record.get('user_idx')} has no conversations")
    history = [conv["messages"] for conv in conversations[:-1]]
    prompt = conversations[-1]["messages"][:1]
    if not prompt or prompt[0].get("role") != "user":
        raise ValueError(
            f"user_idx={synth_record.get('user_idx')} last conversation does not "
            f"start with a user turn: {prompt!r}"
        )
    return history, prompt


# ---------------------------------------------------------------------------
# Joiner: synth + responses + ratings -> PersonalizationExample
# ---------------------------------------------------------------------------


def _index_synth(synth_path: Path) -> dict[str, dict]:
    """Map user_id (str) -> raw synthetic-conversations record."""
    return {str(rec["user_idx"]): rec for rec in load_jsonl(synth_path)}


def _index_responses(resp_path: Path) -> dict[str, dict]:
    """Map user_id -> attribute_responses record."""
    return {str(rec["user_id"]): rec for rec in load_jsonl(resp_path)}


def _index_ratings(judging_dir: Path, glob: str) -> dict[tuple[str, str], dict]:
    """Map (split, user_id) -> attribute_response_judging record.

    The judging part files concatenate train and test users; user_ids are
    only unique within a split (test re-uses 0..N-1), so we key on the
    composite (split, user_id) and de-duplicate within a split.
    """
    by_key: dict[tuple[str, str], dict] = {}
    parts = sorted(judging_dir.glob(glob))
    if not parts:
        raise FileNotFoundError(f"No judging part files found at {judging_dir}/{glob}")
    for part in parts:
        for rec in load_jsonl(part):
            split = rec.get("split")
            if split is None:
                raise ValueError(
                    f"Judging record in {part.name} is missing a 'split' field; "
                    f"cannot disambiguate user_id={rec.get('user_id')!r}"
                )
            key = (split, str(rec["user_id"]))
            if key in by_key:
                raise ValueError(
                    f"Duplicate (split={split!r}, user_id={key[1]!r}) across judging "
                    f"part files (also in {part.name})"
                )
            by_key[key] = rec
    return by_key


def _rating_from_dict(d: dict) -> Rating:
    return Rating(
        target_attribute=d["attribute"],
        target_side=d["side"],
        score=float(d["score"]),
        derived=bool(d["derived"]),
    )


def _candidate_from_dicts(resp_entry: dict, rating_entry: dict | None) -> CandidateResponse:
    if rating_entry is not None:
        # Sanity: response text should be byte-identical to the cached one.
        if rating_entry["response"] != resp_entry["response"]:
            raise ValueError(
                f"Rating/response text mismatch for "
                f"({resp_entry['attribute']!r}, {resp_entry['side']!r})"
            )
        ratings = [_rating_from_dict(s) for s in rating_entry["scores"]]
    else:
        ratings = []
    return CandidateResponse(
        attribute=resp_entry["attribute"],
        side=resp_entry["side"],
        response=resp_entry["response"],
        finish_reason_stop=bool(resp_entry.get("finish_reason_stop", True)),
        ratings=ratings,
    )


def join_one_split(
    *,
    split: Literal["train", "test"],
    synth_path: Path,
    responses_path: Path,
    judging_dir: Path | None,
    judging_glob: str = "ratings_part*.jsonl",
    require_responses: bool = True,
    require_ratings: bool = False,
    input_limit: int | None = None,
) -> Iterator[PersonalizationExample]:
    """Yield joined examples for a single split.

    `input_limit` is applied **before** any filtering, mirroring the upstream
    cap used during candidate generation (so user-id selection stays
    consistent between v2 and the cached artifacts).
    """
    synth_index = _index_synth(synth_path)
    if input_limit is not None:
        # `input_limit` is applied on the raw synthetic-conversations order,
        # which is also how generate_attribute_responses.py applies it.
        keep = list(synth_index.keys())[:input_limit]
        synth_index = {k: synth_index[k] for k in keep}

    responses_index = _index_responses(responses_path) if responses_path is not None else {}
    ratings_index = (
        _index_ratings(judging_dir, judging_glob) if judging_dir is not None else {}
    )

    for user_id, synth_rec in synth_index.items():
        resp_rec = responses_index.get(user_id)
        if resp_rec is None:
            if require_responses:
                raise KeyError(f"No cached responses for user_id={user_id!r} in {responses_path}")
            candidates: list[CandidateResponse] = []
        else:
            rating_rec = ratings_index.get((split, user_id))
            if rating_rec is None and require_ratings:
                raise KeyError(
                    f"No cached ratings for (split={split!r}, user_id={user_id!r}) "
                    f"under {judging_dir}"
                )
            rating_by_combo: dict[tuple[str, str], dict] = {}
            if rating_rec is not None:
                rating_by_combo = {
                    (r["gen_attribute"], r["gen_side"]): r for r in rating_rec["ratings"]
                }
            candidates = [
                _candidate_from_dicts(
                    r, rating_by_combo.get((r["attribute"], r["side"]))
                )
                for r in resp_rec["responses"]
            ]

        history, prompt = extract_history_and_prompt(synth_rec)
        yield PersonalizationExample(
            user_id=user_id,
            split=split,
            gt_user_attributes=list(synth_rec.get("rewrite_style_attributes", [])),
            history=history,
            prompt=prompt,
            candidates=candidates,
        )


# ---------------------------------------------------------------------------
# Materialised v2 dataset (one JSONL per split under data/benchmark_v2/<name>/)
# ---------------------------------------------------------------------------


def example_to_dict(ex: PersonalizationExample) -> dict:
    return {
        "user_id": ex.user_id,
        "split": ex.split,
        "gt_user_attributes": ex.gt_user_attributes,
        "history": ex.history,
        "prompt": ex.prompt,
        "candidates": [
            {
                "attribute": c.attribute,
                "side": c.side,
                "response": c.response,
                "finish_reason_stop": c.finish_reason_stop,
                "ratings": [
                    {
                        "attribute": r.target_attribute,
                        "side": r.target_side,
                        "score": r.score,
                        "derived": r.derived,
                    }
                    for r in c.ratings
                ],
            }
            for c in ex.candidates
        ],
    }


def example_from_dict(d: dict) -> PersonalizationExample:
    return PersonalizationExample(
        user_id=str(d["user_id"]),
        split=d["split"],
        gt_user_attributes=list(d.get("gt_user_attributes", [])),
        history=d["history"],
        prompt=d["prompt"],
        candidates=[
            CandidateResponse(
                attribute=c["attribute"],
                side=c["side"],
                response=c["response"],
                finish_reason_stop=bool(c.get("finish_reason_stop", True)),
                ratings=[
                    Rating(
                        target_attribute=r["attribute"],
                        target_side=r["side"],
                        score=float(r["score"]),
                        derived=bool(r["derived"]),
                    )
                    for r in c.get("ratings", [])
                ],
            )
            for c in d.get("candidates", [])
        ],
    )


@dataclass
class BenchmarkDataset:
    """Pair of (train, test) example lists, loaded from materialized JSONL."""

    name: str
    attributes: list[str]                       # the K response attributes
    sides: list[Side]                           # usually ["follow", "avoid"]
    train: list[PersonalizationExample] = field(default_factory=list)
    test: list[PersonalizationExample] = field(default_factory=list)

    @classmethod
    def load(cls, materialized_dir: Path) -> "BenchmarkDataset":
        meta_path = materialized_dir / "meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        train = [example_from_dict(d) for d in load_jsonl(materialized_dir / "train.jsonl")]
        test = [example_from_dict(d) for d in load_jsonl(materialized_dir / "test.jsonl")]
        return cls(
            name=meta["name"],
            attributes=list(meta["attributes"]),
            sides=list(meta["sides"]),
            train=train,
            test=test,
        )

    def save(self, materialized_dir: Path) -> None:
        materialized_dir.mkdir(parents=True, exist_ok=True)
        save_jsonl(materialized_dir / "train.jsonl", (example_to_dict(e) for e in self.train))
        save_jsonl(materialized_dir / "test.jsonl", (example_to_dict(e) for e in self.test))
        with open(materialized_dir / "meta.json", "w") as f:
            json.dump(
                {"name": self.name, "attributes": self.attributes, "sides": self.sides},
                f,
                indent=2,
            )

    @property
    def user_id_to_gt(self) -> dict[str, list[dict[str, str]]]:
        """Helper for `PersonalizationAttributeJudge.update_user_id_mapping`."""
        out: dict[str, list[dict[str, str]]] = {}
        for ex in (*self.train, *self.test):
            out[ex.user_id] = ex.gt_user_attributes
        return out
