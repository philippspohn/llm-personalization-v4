"""RAG-advanced personalization.

Train (per train user):
  1. Use the LLM to summarize the user's "vibe" from their history (key).
  2. Pick best/worst cached response by weighted user-score.
  3. Use the LLM to summarize the user's preferences from (prompt, best, worst)
     -- what they like and dislike (value).
  4. Embed the vibe text.
  Save {vibe_embedding, vibe_text, preference_text} per user.

Eval (per test user):
  1. Use the LLM to summarize the user's vibe from history.
  2. Embed the vibe.
  3. Top-k cosine retrieval over the train index.
  4. Prepend the K matching preference descriptions to the system prompt.
  5. Generate a personalized response.

The LLM is the gen model (e.g. Qwen3.5-27B). Vibe + preference summarization
share the same vLLM session as the response generation -- we (a) load LLM,
generate vibes + prefs, unload; (b) load embedder, encode, unload; (c) at
eval time: load LLM, generate test vibes; unload, encode, retrieve; reload
LLM, generate responses, unload.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import torch

from llm_personalization.llm.llm_helper import LLMHelper
from llm_personalization.utils.gpu_monitor import log_gpu_usage

from ..cache import CachedDataset
from ..prompts import format_history
from .base import PersonalizationSystemV3
from .rag_simple import _pick_best_worst


VIBE_SYSTEM_PROMPT = """\
You analyze users by their writing style. You will be shown a few prompts a \
user has written. In 2-3 sentences, describe this user's vibe -- their tone, \
interests, level of expertise, and the kind of response style they would \
likely appreciate. Be specific and concrete; avoid generic statements.\
"""

VIBE_USER_TEMPLATE = """\
User's prompts:

{history}

Now write a 2-3 sentence vibe description for this user.\
"""

PREF_SYSTEM_PROMPT = """\
You analyze user preferences. You will be shown a prompt and two responses \
to it: one this user strongly liked, one they strongly disliked. In 2-3 \
sentences, describe what response style this user prefers and what they \
dislike, focusing on the concrete stylistic differences between the two \
responses (tone, length, structure, level of detail, phrasing). Do not \
mention "Response A" or "Response B" -- describe the user's preferences \
in general terms.\
"""

PREF_USER_TEMPLATE = """\
Prompt:
{prompt}

Response the user LIKED:
{best}

Response the user DISLIKED:
{worst}

Now write a 2-3 sentence preference description for this user.\
"""

RAG_SYSTEM_PROMPT = """\
The current user has the following style preferences (inferred from similar users):

{preferences}

Respond to the user's prompt in a style that matches these preferences. Do \
not mention or refer to the preference descriptions above.\
"""


class RAGAdvancedSystem(PersonalizationSystemV3):
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
        self.llm_helper_config = llm_helper_config
        self.llm_helper = LLMHelper(**llm_helper_config)
        self.attributes = list(attributes)
        self.top_k = top_k
        self.embedder_batch_size = embedder_batch_size
        self.embedder_max_seq_length = embedder_max_seq_length

    # ------------------------- helpers -------------------------

    def _load_embedder(self):
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedder = SentenceTransformer(self.embedder_model, device=device, trust_remote_code=True)
        if self.embedder_max_seq_length is not None:
            embedder.max_seq_length = self.embedder_max_seq_length
        return embedder

    def _release_embedder(self, embedder) -> None:
        try:
            embedder.to("cpu")
        except Exception:
            pass
        del embedder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _encode(self, embedder, texts: list[str]) -> np.ndarray:
        emb = embedder.encode(
            texts,
            batch_size=self.embedder_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return emb.astype(np.float32)

    @staticmethod
    def _format_user_prompt(current_messages: list[dict]) -> str:
        return current_messages[0]["content"] if current_messages else ""

    def _build_vibe_prompts(self, histories: list[list[str]]) -> list[list[dict]]:
        return [
            [
                {"role": "system", "content": VIBE_SYSTEM_PROMPT},
                {"role": "user", "content": VIBE_USER_TEMPLATE.format(history=format_history(h))},
            ]
            for h in histories
        ]

    def _build_pref_prompts(
        self, prompts: list[str], bests: list[str], worsts: list[str]
    ) -> list[list[dict]]:
        return [
            [
                {"role": "system", "content": PREF_SYSTEM_PROMPT},
                {"role": "user", "content": PREF_USER_TEMPLATE.format(prompt=p, best=b, worst=w)},
            ]
            for p, b, w in zip(prompts, bests, worsts)
        ]

    # ------------------------- train -------------------------

    def train(
        self,
        dataset: CachedDataset,
        user_id_to_weighted_attrs: dict[str, list[dict]],
        save_path: Path,
        val_dataset: CachedDataset | None = None,
        val_user_id_to_weighted_attrs: dict[str, list[dict]] | None = None,
    ) -> None:
        # val ignored — RAG-advanced has no trainable parameters to validate.
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        print("[RAGAdvancedSystem] 1. Selecting best/worst response per train user...")
        index_users = []
        bests, worsts, prompts_text, histories = [], [], [], []
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
            prompts_text.append(self._format_user_prompt(user.current_messages))
            histories.append(user.history)
        print(f"[RAGAdvancedSystem]    indexed {len(index_users)} users (skipped {skipped})")

        print("[RAGAdvancedSystem] 2. Generating vibe + preference summaries via LLM (batched)...")
        log_gpu_usage("Before LLM load")
        self.llm_helper.load()
        vibe_prompts = self._build_vibe_prompts(histories)
        pref_prompts = self._build_pref_prompts(prompts_text, bests, worsts)
        # Single batched call so vLLM can interleave both kinds of prompts
        all_prompts = vibe_prompts + pref_prompts
        all_responses = self.llm_helper.generate(all_prompts)
        n = len(vibe_prompts)
        vibe_texts = [r.content for r in all_responses[:n]]
        pref_texts = [r.content for r in all_responses[n:]]
        self.llm_helper.unload()
        log_gpu_usage("After LLM unload")

        print("[RAGAdvancedSystem] 3. Encoding vibe descriptions...")
        log_gpu_usage("Before embedder load")
        embedder = self._load_embedder()
        embeddings = self._encode(embedder, vibe_texts)
        self._release_embedder(embedder)
        log_gpu_usage("After embedder unload")

        print(f"[RAGAdvancedSystem] 4. Saving index ({embeddings.shape}) to {save_path}")
        np.save(save_path / "embeddings.npy", embeddings)
        with open(save_path / "meta.jsonl", "w") as f:
            for u, vibe, pref in zip(index_users, vibe_texts, pref_texts):
                f.write(json.dumps({
                    "user_id": u.user_id,
                    "vibe": vibe,
                    "preference": pref,
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
        embeddings = np.load(load_path / "embeddings.npy")
        meta = []
        with open(load_path / "meta.jsonl") as f:
            for line in f:
                meta.append(json.loads(line))
        print(f"[RAGAdvancedSystem] loaded index: {embeddings.shape}, {len(meta)} entries")

        print("[RAGAdvancedSystem] 1. Generating test vibe descriptions via LLM...")
        log_gpu_usage("Before LLM load (vibe)")
        self.llm_helper.load()
        vibe_prompts = self._build_vibe_prompts([u.history for u in dataset])
        test_vibes = [r.content for r in self.llm_helper.generate(vibe_prompts)]
        self.llm_helper.unload()
        log_gpu_usage("After LLM unload (vibe)")

        print("[RAGAdvancedSystem] 2. Encoding + retrieving top-k...")
        log_gpu_usage("Before embedder load")
        embedder = self._load_embedder()
        test_emb = self._encode(embedder, test_vibes)
        self._release_embedder(embedder)
        log_gpu_usage("After embedder unload")

        sims = test_emb @ embeddings.T
        topk_idx = np.argpartition(-sims, kth=min(self.top_k, sims.shape[1] - 1), axis=1)[:, : self.top_k]
        for i in range(topk_idx.shape[0]):
            order = np.argsort(-sims[i, topk_idx[i]])
            topk_idx[i] = topk_idx[i][order]

        print("[RAGAdvancedSystem] 3. Building augmented prompts + generating responses...")
        gen_prompts = []
        for i, user in enumerate(dataset):
            preferences = "\n".join(
                f"- {meta[j]['preference']}" for j in topk_idx[i]
            )
            sys_prompt = RAG_SYSTEM_PROMPT.format(preferences=preferences)
            gen_prompts.append([{"role": "system", "content": sys_prompt}] + user.current_messages)

        log_gpu_usage("Before LLM load (gen)")
        self.llm_helper.load()
        responses = self.llm_helper.generate(gen_prompts)
        self.llm_helper.unload()
        log_gpu_usage("After LLM unload (gen)")
        return [r.content for r in responses]
