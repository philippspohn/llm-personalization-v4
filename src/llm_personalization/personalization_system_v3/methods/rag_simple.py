"""RAG-simple personalization.

Train: for each train user, encode their formatted history with a sentence
transformer; pick the cached candidate response with the highest weighted
user-simulator score as the "best" exemplar and the lowest as the "worst".
Save (embedding, current_prompt, best_response, worst_response) per user.

Eval: encode each test user's history, look up the top-k most similar train
users by cosine similarity, prepend their (prompt, best, worst) tuples to the
system prompt, then run a single batched generation.
"""
from __future__ import annotations

import gc
import json
import math
from pathlib import Path

import numpy as np
import torch

from llm_personalization.llm.llm_helper import LLMHelper
from llm_personalization.utils.gpu_monitor import log_gpu_usage

from ..cache import CachedDataset, CachedUser
from ..prompts import format_history
from .base import PersonalizationSystemV3
from .routing import _weighted_user_score


RAG_SYSTEM_PROMPT = """\
Below are examples of prompts and responses showing the response styles \
that users similar to the current user prefer ("best") and dislike ("worst"). \
Use these examples to infer the kind of response style this user wants, then \
respond accordingly.

<examples>
{examples}
</examples>

Now respond to the user's prompt in a style consistent with what they appear \
to prefer based on the examples above. Do not mention or refer to the examples.
"""

EXAMPLE_TEMPLATE = """\
<example>
<prompt>{prompt}</prompt>
<best_response>{best}</best_response>
<worst_response>{worst}</worst_response>
</example>"""


def _pick_best_worst(user: CachedUser, weighted_targets: list[dict]) -> tuple[str, str] | None:
    """Return (best_response, worst_response) for this user, or None if no
    cached scores are usable."""
    best_attr_side, best_score = None, -math.inf
    worst_attr_side, worst_score = None, math.inf
    for (attr, side) in user.responses.keys():
        s = _weighted_user_score(user, attr, side, weighted_targets)
        if s is None:
            continue
        if s > best_score:
            best_attr_side, best_score = (attr, side), s
        if s < worst_score:
            worst_attr_side, worst_score = (attr, side), s
    if best_attr_side is None or worst_attr_side is None or best_attr_side == worst_attr_side:
        return None
    return user.responses[best_attr_side], user.responses[worst_attr_side]


class RAGSimpleSystem(PersonalizationSystemV3):
    def __init__(
        self,
        embedder_model: str,
        llm_helper_config: dict,
        attributes: list[str],
        top_k: int = 3,
        embedder_batch_size: int = 32,
        embedder_max_seq_length: int | None = None,
    ):
        self.embedder_model = embedder_model
        self.llm_helper = LLMHelper(**llm_helper_config)
        self.attributes = list(attributes)
        self.top_k = top_k
        self.embedder_batch_size = embedder_batch_size
        self.embedder_max_seq_length = embedder_max_seq_length

    # ------------------------- helpers -------------------------

    def _load_embedder(self):
        # Lazy import: only required when this method is actually used
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedder = SentenceTransformer(self.embedder_model, device=device, trust_remote_code=True)
        if self.embedder_max_seq_length is not None:
            embedder.max_seq_length = self.embedder_max_seq_length
        return embedder

    def _encode(self, embedder, texts: list[str]) -> np.ndarray:
        emb = embedder.encode(
            texts,
            batch_size=self.embedder_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return emb.astype(np.float32)

    def _release_embedder(self, embedder) -> None:
        """Aggressively free GPU memory held by the sentence-transformer.
        `del + empty_cache` alone leaves ~3 GiB pinned, which is enough to OOM
        a 27B model load right after."""
        try:
            embedder.to("cpu")
        except Exception:
            pass
        del embedder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @staticmethod
    def _format_user_prompt(current_messages: list[dict]) -> str:
        # current_messages is `[{"role": "user", "content": ...}]`
        return current_messages[0]["content"] if current_messages else ""

    # ------------------------- train -------------------------

    def train(
        self,
        dataset: CachedDataset,
        user_id_to_weighted_attrs: dict[str, list[dict]],
        save_path: Path,
        val_dataset: CachedDataset | None = None,
        val_user_id_to_weighted_attrs: dict[str, list[dict]] | None = None,
    ) -> None:
        # val ignored — RAG-simple has no trainable parameters to validate.
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        print("[RAGSimpleSystem] 1. Selecting best/worst response per train user...")
        index_users: list[CachedUser] = []
        bests: list[str] = []
        worsts: list[str] = []
        skipped = 0
        for user in dataset:
            targets = user_id_to_weighted_attrs.get(user.user_id, [])
            if not targets:
                skipped += 1
                continue
            picked = _pick_best_worst(user, targets)
            if picked is None:
                skipped += 1
                continue
            best, worst = picked
            index_users.append(user)
            bests.append(best)
            worsts.append(worst)
        print(f"[RAGSimpleSystem]    indexed {len(index_users)} users (skipped {skipped})")

        print("[RAGSimpleSystem] 2. Encoding histories...")
        log_gpu_usage("Before embedder load")
        embedder = self._load_embedder()
        log_gpu_usage("After embedder load")
        history_texts = [format_history(u.history) for u in index_users]
        embeddings = self._encode(embedder, history_texts)
        self._release_embedder(embedder)
        log_gpu_usage("After embedder unload")

        print(f"[RAGSimpleSystem] 3. Saving index ({embeddings.shape}) to {save_path}")
        np.save(save_path / "embeddings.npy", embeddings)
        with open(save_path / "meta.jsonl", "w") as f:
            for u, best, worst in zip(index_users, bests, worsts):
                f.write(json.dumps({
                    "user_id": u.user_id,
                    "current_prompt": self._format_user_prompt(u.current_messages),
                    "best_response": best,
                    "worst_response": worst,
                }) + "\n")
        with open(save_path / "config.json", "w") as f:
            json.dump({
                "embedder_model": self.embedder_model,
                "top_k": self.top_k,
                "n_index_users": len(index_users),
            }, f, indent=2)

    # ------------------------- evaluate -------------------------

    def evaluate(
        self,
        dataset: CachedDataset,
        load_path: Path,
        user_id_to_weighted_attrs: dict[str, list[dict]] | None = None,
    ) -> list[str]:
        load_path = Path(load_path)
        embeddings = np.load(load_path / "embeddings.npy")  # (N_train, D), already normalized
        meta = []
        with open(load_path / "meta.jsonl") as f:
            for line in f:
                meta.append(json.loads(line))
        print(f"[RAGSimpleSystem] loaded index: {embeddings.shape}, {len(meta)} entries")

        print("[RAGSimpleSystem] 1. Encoding test histories...")
        log_gpu_usage("Before embedder load")
        embedder = self._load_embedder()
        test_emb = self._encode(embedder, [format_history(u.history) for u in dataset])
        self._release_embedder(embedder)
        log_gpu_usage("After embedder unload")

        print("[RAGSimpleSystem] 2. Top-k retrieval (cosine, normalized)...")
        # both already L2-normalized -> dot product == cosine
        sims = test_emb @ embeddings.T  # (N_test, N_train)
        topk_idx = np.argpartition(-sims, kth=min(self.top_k, sims.shape[1] - 1), axis=1)[:, : self.top_k]
        # Sort the top-k slice by descending sim for nicer prompts
        for i in range(topk_idx.shape[0]):
            order = np.argsort(-sims[i, topk_idx[i]])
            topk_idx[i] = topk_idx[i][order]

        print("[RAGSimpleSystem] 3. Building augmented prompts...")
        prompts = []
        for i, user in enumerate(dataset):
            examples = "\n".join(
                EXAMPLE_TEMPLATE.format(
                    prompt=meta[j]["current_prompt"],
                    best=meta[j]["best_response"],
                    worst=meta[j]["worst_response"],
                )
                for j in topk_idx[i]
            )
            sys_prompt = RAG_SYSTEM_PROMPT.format(examples=examples)
            prompts.append([{"role": "system", "content": sys_prompt}] + user.current_messages)

        print("[RAGSimpleSystem] 4. Generating responses (single batched vLLM call)...")
        log_gpu_usage("Before LLM load")
        self.llm_helper.load()
        responses = self.llm_helper.generate(prompts)
        self.llm_helper.unload()
        log_gpu_usage("After LLM unload")
        return [r.content for r in responses]
