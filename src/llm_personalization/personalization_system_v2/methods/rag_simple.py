"""Simple RAG personalization.

For each train user with cached ratings:
  - encode their conversation history with a frozen sentence encoder
  - find the candidate that best matches their GT axis (= preferred response)
    and the candidate that matches the GT axis least (= disliked response)
  - store (embedding, prompt_text, preferred_response, disliked_response)

At test time:
  - encode the test user's conversation history
  - top-`k` nearest train users by cosine similarity
  - build a system prompt that lists the k retrieved
    (prompt, preferred, disliked) triples as in-context examples
  - single batched vLLM call produces the personalized response

No LLM calls during fit; one batched LLM call at generate time.
"""

from __future__ import annotations

import gc

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from llm_personalization.llm.llm_helper import LLMHelper

from ..dataset import CandidateResponse, PersonalizationExample, Side
from ..method import MethodInput, PersonalizationMethod


PREFERENCE_HEADER = (
    "You will respond to the user's prompt. To help you tailor the response "
    "style, here are {k} examples from similar users showing a response they "
    "preferred and a response they disliked. Match the style of the preferred "
    "responses; avoid the style of the disliked ones. Do NOT copy the example "
    "content -- only mirror the style.\n"
)

EXAMPLE_TEMPLATE = (
    "<example {i}>\n"
    "<user_prompt>{prompt}</user_prompt>\n"
    "<preferred_response>{preferred}</preferred_response>\n"
    "<disliked_response>{disliked}</disliked_response>\n"
    "</example {i}>\n"
)


def _flatten_history_text(
    history: list[list[dict[str, str]]],
    prompt: list[dict[str, str]],
    max_chars_per_msg: int,
) -> str:
    parts: list[str] = []
    for conv in history:
        for msg in conv:
            content = msg["content"]
            if len(content) > max_chars_per_msg:
                content = content[:max_chars_per_msg] + "..."
            parts.append(f"[{msg['role']}] {content}")
    for msg in prompt:
        content = msg["content"]
        if len(content) > max_chars_per_msg:
            content = content[:max_chars_per_msg] + "..."
        parts.append(f"[{msg['role']}] {content}")
    return "\n".join(parts)


def _pick_best_worst(
    example: PersonalizationExample,
) -> tuple[CandidateResponse, CandidateResponse] | None:
    """Best/worst candidate by judge rating against the user's first GT axis.

    Returns None if the example lacks GT or full ratings.
    """
    if not example.gt_user_attributes or not example.has_ratings:
        return None
    gt_attr = example.gt_user_attributes[0]["attribute"]
    gt_side: Side = example.gt_user_attributes[0]["side"]

    scored: list[tuple[float, CandidateResponse]] = []
    for c in example.candidates:
        rating = next(
            (r for r in c.ratings if r.target_attribute == gt_attr and r.target_side == gt_side),
            None,
        )
        if rating is None:
            return None
        scored.append((rating.score, c))
    scored.sort(key=lambda t: t[0])
    return scored[-1][1], scored[0][1]


class RAGSimple(PersonalizationMethod):
    needs_gt_at_test = False

    def __init__(
        self,
        llm: LLMHelper,
        embedding_model: str = "BAAI/bge-m3",
        embedding_max_length: int = 1024,
        embedding_batch_size: int = 32,
        max_chars_per_msg: int = 1024,
        k: int = 3,
    ):
        self.llm = llm
        self.embedding_model_name = embedding_model
        self.embedding_max_length = embedding_max_length
        self.embedding_batch_size = embedding_batch_size
        self.max_chars_per_msg = max_chars_per_msg
        self.k = k

        self._embed_tokenizer: AutoTokenizer | None = None
        self._embed_model: AutoModel | None = None

        self._train_embeddings: np.ndarray | None = None
        self._train_prompts: list[str] = []
        self._train_preferred: list[str] = []
        self._train_disliked: list[str] = []

    # ---------------------------------------------------------- embedding
    def _load_embedder(self):
        self._embed_tokenizer = AutoTokenizer.from_pretrained(
            self.embedding_model_name, trust_remote_code=True
        )
        self._embed_model = AutoModel.from_pretrained(
            self.embedding_model_name, trust_remote_code=True
        )
        self._embed_model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._embed_model.to(device)

    def _unload_embedder(self):
        if self._embed_model is None:
            return
        del self._embed_model
        del self._embed_tokenizer
        self._embed_model = None
        self._embed_tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def _encode(self, texts: list[str]) -> np.ndarray:
        assert self._embed_model is not None and self._embed_tokenizer is not None
        out: list[np.ndarray] = []
        for i in range(0, len(texts), self.embedding_batch_size):
            batch = texts[i : i + self.embedding_batch_size]
            inputs = self._embed_tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.embedding_max_length,
                return_tensors="pt",
            ).to(self._embed_model.device)
            outputs = self._embed_model(**inputs)
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            pooled = (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            out.append(pooled.cpu().numpy())
        return np.concatenate(out, axis=0)

    # ----------------------------------------------------------------- fit
    def fit(self, train: list[PersonalizationExample]) -> None:
        kept_examples: list[PersonalizationExample] = []
        prompts: list[str] = []
        preferred: list[str] = []
        disliked: list[str] = []

        for ex in train:
            picked = _pick_best_worst(ex)
            if picked is None:
                continue
            best, worst = picked
            prompts.append(ex.prompt[0]["content"] if ex.prompt else "")
            preferred.append(best.response)
            disliked.append(worst.response)
            kept_examples.append(ex)

        if not kept_examples:
            raise RuntimeError("No usable training examples for RAGSimple (need GT + ratings)")

        print(
            f"[RAGSimple] Indexing {len(kept_examples)} train users "
            f"(dropped {len(train) - len(kept_examples)})",
            flush=True,
        )

        history_texts = [
            _flatten_history_text(ex.history, ex.prompt, self.max_chars_per_msg)
            for ex in kept_examples
        ]

        print(f"[RAGSimple] Loading embedding model {self.embedding_model_name!r}...", flush=True)
        self._load_embedder()
        try:
            print(f"[RAGSimple] Encoding train histories...", flush=True)
            self._train_embeddings = self._encode(history_texts)
        finally:
            # We DON'T unload yet: we still need to encode the test side at
            # generate(). The orchestrator will call generate() right after.
            pass

        self._train_prompts = prompts
        self._train_preferred = preferred
        self._train_disliked = disliked
        print(f"[RAGSimple] Fit done. embeddings.shape = {self._train_embeddings.shape}", flush=True)

    # ----------------------------------------------------------- generate
    def generate(self, batch: list[MethodInput]) -> list[str]:
        if self._train_embeddings is None:
            raise RuntimeError("RAGSimple.fit must be called before generate")

        # Encode the test side (embedder is already loaded from fit; load if not).
        if self._embed_model is None:
            self._load_embedder()

        test_history_texts = [
            _flatten_history_text(item.history, item.prompt, self.max_chars_per_msg)
            for item in batch
        ]
        print(f"[RAGSimple] Encoding {len(test_history_texts)} test histories...", flush=True)
        test_embeddings = self._encode(test_history_texts)
        self._unload_embedder()                       # free GPU before vLLM loads

        # Cosine similarity (both already L2-normalized).
        sims = test_embeddings @ self._train_embeddings.T   # (n_test, n_train)

        # Build personalized system prompts.
        sys_prompts: list[str] = []
        for i in range(len(batch)):
            top_idx = np.argpartition(-sims[i], self.k)[: self.k]
            top_idx = top_idx[np.argsort(-sims[i, top_idx])]
            example_strs = [
                EXAMPLE_TEMPLATE.format(
                    i=j + 1,
                    prompt=self._train_prompts[idx],
                    preferred=self._train_preferred[idx],
                    disliked=self._train_disliked[idx],
                )
                for j, idx in enumerate(top_idx)
            ]
            sys_prompts.append(PREFERENCE_HEADER.format(k=self.k) + "\n".join(example_strs))

        # Diagnostic
        for i in range(min(2, len(batch))):
            print(
                f"[RAGSimple] test user {batch[i].user_id} -> top-{self.k} similarities: "
                f"{sorted(sims[i], reverse=True)[: self.k]}",
                flush=True,
            )

        prompts = [
            [{"role": "system", "content": sp}, *item.prompt]
            for sp, item in zip(sys_prompts, batch)
        ]

        self.llm.load()
        try:
            responses = self.llm.generate(prompts)
        finally:
            self.llm.unload()
        return [r.content for r in responses]
