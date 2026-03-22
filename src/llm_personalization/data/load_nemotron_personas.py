from typing import Literal
from datasets import load_dataset
import random

TRAIN_PREFIXES = tuple("0123456789")
TEST_PREFIXES = tuple("abcdef")


def load_nemotron_personas(
    split: Literal["train", "test"],
    limit: int,
    seed: int | None = None,
) -> list[dict[str, str]]:
    """Load personas from nvidia/Nemotron-Personas-USA.

    Uses uuid prefix for train/test split (0-9 → train, a-f → test).
    Only streams as many rows as needed.
    Returns dicts with keys: persona, skills_and_expertise, hobbies_and_interests.
    """
    prefixes = TRAIN_PREFIXES if split == "train" else TEST_PREFIXES
    ds = load_dataset("nvidia/Nemotron-Personas-USA", split="train", streaming=True)

    personas = []
    for row in ds:
        if not row["uuid"].startswith(prefixes):
            continue
        personas.append({
            "uuid": row["uuid"],
            "persona": row["persona"],
            "skills_and_expertise": row["skills_and_expertise"],
            "hobbies_and_interests": row["hobbies_and_interests"],
        })
        if len(personas) >= limit:
            break

    if seed is not None:
        random.Random(seed).shuffle(personas)

    return personas
