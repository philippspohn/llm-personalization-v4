from llm_personalization.benchmark.personalization_system import PersonalizationSystem, PersonalizationDataset
from llm_personalization.benchmark.personalization_judge import PersonalizationJudge
from llm_personalization.personalization_system.attribute_personalization.attribute_personalization_system import _format_system_prompt
from llm_personalization.llm.llm_helper import LLMHelper
from llm_personalization.utils.gpu_monitor import log_gpu_usage
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from typing import Any
import json
import torch
import numpy as np
import gc


STYLE_DESCRIPTOR_TEMPLATE = """Look at the following conversations. Create a concise description of the response style this user prefers. Focus on tone, detail level, formality, structure, and approach.

{examples}

Based on these examples, describe in 2-3 sentences what response style this user prefers. Only one description that encompasses all the examples. Output only the description, nothing else."""

STYLE_EXAMPLE_TEMPLATE = """Example {i}:
USER PROMPT: {prompt}
PREFERRED RESPONSE: {liked_response}
DISLIKED RESPONSE: {disliked_response}
"""

PERSONALIZED_SYSTEM_PROMPT = """Respond to the user's message using the following style preferences:
{style_descriptor}"""


class RAGPersonalizationSystem(PersonalizationSystem):
    def __init__(self,
        llm_helper_config: dict,
        attributes: list[str],
        embedding_model: str = "BAAI/bge-m3",
        embedding_max_length: int = 512,
        embedding_batch_size: int = 64,
        k: int = 4,
        sampling_params_style: dict[str, Any] | None = None,
    ):
        self.llm_helper_config = llm_helper_config
        self.llm_helper = LLMHelper(**self.llm_helper_config)
        self.attributes = attributes
        self.embedding_model_name = embedding_model
        self.embedding_max_length = embedding_max_length
        self.embedding_batch_size = embedding_batch_size
        self.k = k
        self.sampling_params_style = sampling_params_style or {"max_tokens": 256}

    def _load_embedding_model(self):
        self.embed_tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name, trust_remote_code=True)
        self.embed_model = AutoModel.from_pretrained(self.embedding_model_name, trust_remote_code=True)
        self.embed_model.eval()
        self.embed_model.to("cuda" if torch.cuda.is_available() else "cpu")

    def _unload_embedding_model(self):
        if hasattr(self, "embed_model") and self.embed_model is not None:
            del self.embed_model
            del self.embed_tokenizer
            self.embed_model = None
            self.embed_tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), self.embedding_batch_size):
            batch = texts[i:i + self.embedding_batch_size]
            inputs = self.embed_tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.embedding_max_length, return_tensors="pt",
            ).to(self.embed_model.device)
            with torch.no_grad():
                outputs = self.embed_model(**inputs)
            # Mean pooling over non-padding tokens
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            pooled = (token_embeddings * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)
            # L2 normalize
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            all_embeddings.append(pooled.cpu().numpy())
        return np.concatenate(all_embeddings, axis=0)

    @staticmethod
    def _extract_prompt_text(current_messages: list[dict[str, str]]) -> str:
        """Extract the user prompt text from current_messages."""
        for msg in current_messages:
            if msg["role"] == "user":
                return msg["content"]
        return current_messages[0]["content"]

    def train(self, dataset: PersonalizationDataset, judge: PersonalizationJudge, save_path: Path, gt_user_attributes: dict[str, list[dict[str, str]]] | None = None):
        save_path.mkdir(parents=True, exist_ok=True)

        # 1. Generate attribute-conditioned responses for each item
        print("[RAGPersonalizationSystem] 1. Generating attribute-conditioned responses...")
        generation_prompts = []
        generation_metadata = []  # (item_idx, attribute)
        for i, item in enumerate(dataset):
            for attribute in self.attributes:
                generation_prompts.append([
                    {"role": "system", "content": _format_system_prompt(attribute, "follow")},
                ] + item.current_messages)
                generation_metadata.append({"item_idx": i, "attribute": attribute})

        log_gpu_usage("Before LLM load")
        self.llm_helper.load()
        log_gpu_usage("After LLM load")
        responses = self.llm_helper.generate(generation_prompts)
        self.llm_helper.unload()
        log_gpu_usage("After LLM unload")

        # 2. Judge all responses
        print("[RAGPersonalizationSystem] 2. Judging responses...")
        judge_user_ids = []
        judge_conversations = []
        for meta, response in zip(generation_metadata, responses):
            item = dataset[meta["item_idx"]]
            judge_user_ids.append(item.user_id)
            judge_conversations.append(
                item.current_messages + [{"role": "assistant", "content": response.content}]
            )

        log_gpu_usage("Before judge load")
        judge.load()
        log_gpu_usage("After judge load")
        judge_scores = judge.judge(judge_user_ids, judge_conversations)
        judge.unload()
        log_gpu_usage("After judge unload")

        # 3. Build per-item response store
        print("[RAGPersonalizationSystem] 3. Building response store...")
        num_items = len(dataset)
        num_attrs = len(self.attributes)
        store = []
        for i in range(num_items):
            item = dataset[i]
            prompt_text = self._extract_prompt_text(item.current_messages)
            item_responses = []
            for j, attribute in enumerate(self.attributes):
                idx = i * num_attrs + j
                item_responses.append({
                    "attribute": attribute,
                    "response": responses[idx].content,
                    "score": judge_scores[idx],
                })
            store.append({
                "user_id": item.user_id,
                "prompt_text": prompt_text,
                "responses": item_responses,
            })

        # Debug: print a few examples
        for s in store[:3]:
            scores_summary = {r["attribute"]: r["score"] for r in s["responses"]}
            best = max(s["responses"], key=lambda r: r["score"])
            worst = min(s["responses"], key=lambda r: r["score"])
            print(f"[RAGPersonalizationSystem]   User {s['user_id']}: best={best['attribute']} ({best['score']:.2f}), worst={worst['attribute']} ({worst['score']:.2f})")
            print(f"[RAGPersonalizationSystem]   All scores: {scores_summary}")

        # 4. Encode prompts
        print("[RAGPersonalizationSystem] 4. Encoding prompts...")
        prompt_texts = [s["prompt_text"] for s in store]
        self._load_embedding_model()
        embeddings = self._encode_texts(prompt_texts)
        self._unload_embedding_model()

        # 5. Save
        print("[RAGPersonalizationSystem] 5. Saving...")
        np.save(save_path / "embeddings.npy", embeddings)
        with open(save_path / "store.json", "w") as f:
            json.dump(store, f)
        with open(save_path / "config.json", "w") as f:
            json.dump({
                "embedding_model": self.embedding_model_name,
                "embedding_max_length": self.embedding_max_length,
                "k": self.k,
                "attributes": list(self.attributes),
            }, f)

        print(f"[RAGPersonalizationSystem] Saved {len(store)} items to {save_path}")

    def evaluate(self, dataset: PersonalizationDataset, load_path: Path, **kwargs) -> list[str]:
        # 1. Load stored data
        print("[RAGPersonalizationSystem] 1. Loading stored data...")
        embeddings = np.load(load_path / "embeddings.npy")
        with open(load_path / "store.json", "r") as f:
            store = json.load(f)
        with open(load_path / "config.json", "r") as f:
            config = json.load(f)

        # 2. Encode test prompts
        print("[RAGPersonalizationSystem] 2. Encoding test prompts...")
        test_prompt_texts = [self._extract_prompt_text(dataset[i].current_messages) for i in range(len(dataset))]
        self._load_embedding_model()
        test_embeddings = self._encode_texts(test_prompt_texts)
        self._unload_embedding_model()

        # 3. Retrieve k nearest neighbors for each test prompt
        print("[RAGPersonalizationSystem] 3. Retrieving nearest neighbors...")
        k = config["k"]
        # Cosine similarity (embeddings are already L2-normalized)
        similarities = test_embeddings @ embeddings.T  # (num_test, num_train)

        retrieved_examples = []
        for i in range(len(dataset)):
            top_k_indices = np.argsort(similarities[i])[-k:][::-1]
            examples = []
            for idx in top_k_indices:
                entry = store[idx]
                best = max(entry["responses"], key=lambda r: r["score"])
                worst = min(entry["responses"], key=lambda r: r["score"])
                examples.append({
                    "prompt": entry["prompt_text"],
                    "liked_response": best["response"],
                    "disliked_response": worst["response"],
                    "similarity": float(similarities[i, idx]),
                })
            retrieved_examples.append(examples)

        # Debug: print retrieval for first few items
        for i in range(min(3, len(retrieved_examples))):
            print(f"[RAGPersonalizationSystem]   Test item {i}: query='{test_prompt_texts[i][:80]}...'")
            for j, ex in enumerate(retrieved_examples[i]):
                print(f"    Neighbor {j}: sim={ex['similarity']:.4f}, prompt='{ex['prompt'][:60]}...'")

        # 4. Generate style descriptors
        print("[RAGPersonalizationSystem] 4. Generating style descriptors...")
        style_prompts = []
        for examples in retrieved_examples:
            example_strs = []
            for j, ex in enumerate(examples):
                example_strs.append(STYLE_EXAMPLE_TEMPLATE.format(
                    i=j + 1,
                    prompt=ex["prompt"],
                    liked_response=ex["liked_response"],
                    disliked_response=ex["disliked_response"],
                ))
            style_prompt = STYLE_DESCRIPTOR_TEMPLATE.format(examples="\n".join(example_strs))
            style_prompts.append([{"role": "user", "content": style_prompt}])

        log_gpu_usage("Before LLM load (style)")
        self.llm_helper.load()

        # Use style-specific sampling params if provided
        original_params = self.llm_helper.sampling_params
        self.llm_helper.sampling_params = self.sampling_params_style
        style_responses = self.llm_helper.generate(style_prompts)
        self.llm_helper.sampling_params = original_params

        style_descriptors = [r.content for r in style_responses]

        # Debug: print first few style descriptors
        for i in range(min(3, len(style_descriptors))):
            print(f"[RAGPersonalizationSystem]   Style descriptor {i}: {style_descriptors[i][:200]}")

        # 5. Generate personalized responses
        print("[RAGPersonalizationSystem] 5. Generating personalized responses...")
        generation_prompts = []
        for i in range(len(dataset)):
            item = dataset[i]
            system_prompt = PERSONALIZED_SYSTEM_PROMPT.format(style_descriptor=style_descriptors[i])
            generation_prompts.append([
                {"role": "system", "content": system_prompt},
            ] + item.current_messages)

        responses = self.llm_helper.generate(generation_prompts)
        self.llm_helper.unload()
        log_gpu_usage("After LLM unload")

        return [r.content for r in responses]
