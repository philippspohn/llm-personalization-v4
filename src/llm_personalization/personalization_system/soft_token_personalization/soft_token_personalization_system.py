"""
Soft-Token Personalization System.

Training:
    1. Generate attribute-conditioned responses (follow + avoid sides) with the
       big LLM helper, as in AttributePersonalizationSystem.
    2. Judge them with the personalization judge.
    3. For each user, pick highest-scoring response as `chosen` and
       lowest-scoring as `rejected` -> preference pair (prompt, chosen, rejected).
    4. Encode user conversation history with a frozen text encoder (bge-m3).
    5. Train a Linear projector (encoder_dim -> N * hidden_size) + a LoRA
       adapter on the trainable LLM, jointly via DPO.

Inference:
    Encode the test user's history, project to soft tokens, prepend to the
    LLM's input embeddings, generate with HF .generate().
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from llm_personalization.benchmark.personalization_judge import PersonalizationJudge
from llm_personalization.benchmark.personalization_system import (
    PersonalizationDataset,
    PersonalizationSystem,
)
from llm_personalization.llm.llm_helper import LLMHelper
from llm_personalization.personalization_system.attribute_personalization.attribute_personalization_system import (
    _format_system_prompt,
)
from llm_personalization.personalization_system.soft_token_personalization.soft_token_model import (
    SoftTokenLLM,
    SoftTokenLLMConfig,
)
from llm_personalization.utils.gpu_monitor import log_gpu_usage


def _apply_chat_template_ids(tokenizer, msgs, add_gen_prompt: bool) -> list[int]:
    """Apply chat template and normalize output to a flat list[int] of token ids.

    Newer transformers versions can return a BatchEncoding, dict, str, or nested
    list depending on the tokenizer. This coerces all of those into list[int].
    """
    out = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=add_gen_prompt, tokenize=True,
    )
    if hasattr(out, "input_ids"):
        out = out.input_ids
    elif isinstance(out, dict) and "input_ids" in out:
        out = out["input_ids"]
    if isinstance(out, str):
        enc = tokenizer(out, add_special_tokens=False)
        out = enc["input_ids"] if isinstance(enc, dict) else enc.input_ids
    if len(out) > 0 and isinstance(out[0], (list, tuple)):
        out = out[0]
    return [int(t) for t in out]


def _format_history(history: list[list[dict[str, str]]]) -> str:
    text = ""
    for conversation in history:
        text += "<conversation>\n"
        for message in conversation:
            text += f"<message role='{message['role']}'>{message['content']}</message>\n"
        text += "</conversation>\n"
    return text


class SoftTokenPersonalizationSystem(PersonalizationSystem):
    def __init__(
        self,
        soft_token_model_config: dict,
        generation_llm_helper_config: dict,
        attributes: list[str],
        dpo_train_kwargs: dict | None = None,
        encoder_batch_size: int = 16,
        generation_kwargs: dict | None = None,
    ):
        self.attributes = list(attributes)
        self.soft_token_model_config = dict(soft_token_model_config)
        self.generation_llm_helper_config = dict(generation_llm_helper_config)
        self.generation_llm_helper = LLMHelper(**self.generation_llm_helper_config)
        self.dpo_train_kwargs = dict(dpo_train_kwargs or {})
        self.encoder_batch_size = encoder_batch_size
        self.generation_kwargs = dict(generation_kwargs or {"max_new_tokens": 512, "do_sample": False})
        self._soft_token_model: SoftTokenLLM | None = None

    # -----------------------------------------------------------------------
    # TRAIN
    # -----------------------------------------------------------------------

    def train(
        self,
        dataset: PersonalizationDataset,
        judge: PersonalizationJudge,
        save_path: Path,
        gt_user_attributes: dict[str, list[dict[str, str]]] | None = None,
    ):
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # 1. Generate attribute-conditioned responses (follow + avoid) -----------
        print("[SoftTokenPersonalizationSystem] 1. Generating attribute-conditioned responses...")
        generation_prompts: list[list[dict[str, str]]] = []
        generation_metadata: list[dict[str, Any]] = []
        for i, item in enumerate(dataset):
            for attribute in self.attributes:
                for side in ("follow", "avoid"):
                    generation_prompts.append(
                        [{"role": "system", "content": _format_system_prompt(attribute, side)}]
                        + item.current_messages
                    )
                    generation_metadata.append({
                        "item_idx": i,
                        "user_id": item.user_id,
                        "attribute": attribute,
                        "side": side,
                        "current_messages": item.current_messages,
                    })

        log_gpu_usage("Before LLM load")
        self.generation_llm_helper.load()
        log_gpu_usage("After LLM load")
        responses = self.generation_llm_helper.generate(generation_prompts)
        self.generation_llm_helper.unload()
        log_gpu_usage("After LLM unload")

        # 2. Judge responses ------------------------------------------------------
        print("[SoftTokenPersonalizationSystem] 2. Judging responses...")
        judge_user_ids = [m["user_id"] for m in generation_metadata]
        judge_conversations = [
            m["current_messages"] + [{"role": "assistant", "content": r.content}]
            for m, r in zip(generation_metadata, responses)
        ]
        log_gpu_usage("Before judge load")
        judge.load()
        log_gpu_usage("After judge load")
        raw_scores = judge.judge(judge_user_ids, judge_conversations)
        judge.unload()
        log_gpu_usage("After judge unload")

        # 3. Build preference pairs: per user, best vs worst ---------------------
        print("[SoftTokenPersonalizationSystem] 3. Building preference pairs...")
        df = pd.DataFrame([
            {
                "item_idx": m["item_idx"],
                "user_id": m["user_id"],
                "attribute": m["attribute"],
                "side": m["side"],
                "response": r.content,
                "score": s,
            }
            for m, r, s in zip(generation_metadata, responses, raw_scores)
        ])
        # Fill NaN scores with per-user mean so they don't get picked.
        nan_count = df["score"].isna().sum()
        if nan_count > 0:
            df["score"] = df.groupby("user_id")["score"].transform(lambda s: s.fillna(s.mean()))
            df["score"] = df["score"].fillna(0.0)

        preference_pairs: list[dict[str, Any]] = []
        for user_id, group in df.groupby("user_id"):
            if len(group) < 2:
                continue
            best = group.loc[group["score"].idxmax()]
            worst = group.loc[group["score"].idxmin()]
            if best["score"] == worst["score"]:
                continue  # no signal
            item_idx = int(best["item_idx"])
            item = dataset[item_idx]
            preference_pairs.append({
                "user_id": user_id,
                "item_idx": item_idx,
                "current_messages": item.current_messages,
                "history_text": _format_history(item.conversation_history),
                "chosen": best["response"],
                "rejected": worst["response"],
                "chosen_attr": f"{best['attribute']}/{best['side']}",
                "rejected_attr": f"{worst['attribute']}/{worst['side']}",
                "chosen_score": float(best["score"]),
                "rejected_score": float(worst["score"]),
            })

        print(f"[SoftTokenPersonalizationSystem]    Built {len(preference_pairs)} preference pairs")
        for p in preference_pairs[:3]:
            print(
                f"      user={p['user_id']} | chosen={p['chosen_attr']} ({p['chosen_score']:.2f}) "
                f"vs rejected={p['rejected_attr']} ({p['rejected_score']:.2f})"
            )

        # Persist preference-pair metadata for debugging.
        with open(save_path / "preference_pairs.json", "w") as f:
            json.dump(preference_pairs, f)

        # 4. Load soft-token model, encode histories -----------------------------
        print("[SoftTokenPersonalizationSystem] 4. Loading soft-token model and encoding histories...")
        model_config = SoftTokenLLMConfig(**self.soft_token_model_config)
        self._soft_token_model = SoftTokenLLM(model_config)
        self._soft_token_model.load_base(with_lora=True)
        log_gpu_usage("After soft-token model load")

        history_texts = [p["history_text"] for p in preference_pairs]
        context_embeddings = self._soft_token_model.encode_history(
            history_texts, batch_size=self.encoder_batch_size
        )  # (N, enc_dim), fp32 on CPU

        # 5. DPO training --------------------------------------------------------
        print("[SoftTokenPersonalizationSystem] 5. Running DPO training...")
        self._train_dpo(preference_pairs, context_embeddings)

        # 6. Save ----------------------------------------------------------------
        print("[SoftTokenPersonalizationSystem] 6. Saving trained soft-token model...")
        self._soft_token_model.save(save_path)
        with open(save_path / "attributes.json", "w") as f:
            json.dump(self.attributes, f)
        with open(save_path / "soft_token_model_config.json", "w") as f:
            json.dump(self.soft_token_model_config, f)

        self._soft_token_model.unload()
        self._soft_token_model = None
        log_gpu_usage("After soft-token model unload")

    # -----------------------------------------------------------------------
    # DPO
    # -----------------------------------------------------------------------

    def _tokenize_pair(
        self,
        tokenizer,
        messages: list[dict[str, str]],
        response: str,
        max_length: int,
    ) -> dict[str, torch.Tensor]:
        """
        Apply chat template to `messages` (prompt), then append `response` as the
        assistant turn. Return input_ids, attention_mask, labels with prompt
        positions masked out (-100).

        We tokenize prompt and prompt+response separately (both via chat
        template) so we can figure out where the assistant response starts.
        """
        prompt_ids = _apply_chat_template_ids(tokenizer, messages, add_gen_prompt=True)
        full_ids = _apply_chat_template_ids(
            tokenizer,
            messages + [{"role": "assistant", "content": response}],
            add_gen_prompt=False,
        )

        # Defensive: ensure prompt is a prefix. If not, fall back to naive concat.
        if full_ids[: len(prompt_ids)] != prompt_ids:
            resp_enc = tokenizer(response, add_special_tokens=False)
            resp_ids = resp_enc["input_ids"] if isinstance(resp_enc, dict) or hasattr(resp_enc, "input_ids") else resp_enc
            if hasattr(resp_enc, "input_ids"):
                resp_ids = resp_enc.input_ids
            resp_ids = [int(t) for t in resp_ids]
            eos = tokenizer.eos_token_id
            full_ids = list(prompt_ids) + list(resp_ids) + ([eos] if eos is not None else [])

        # Truncate from the left of the prompt if too long, preserving response.
        if len(full_ids) > max_length:
            over = len(full_ids) - max_length
            new_prompt_ids = list(prompt_ids)[over:]
            full_ids = new_prompt_ids + list(full_ids[len(prompt_ids):])
            prompt_len = len(new_prompt_ids)
        else:
            prompt_len = len(prompt_ids)

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def _collate(self, batch: list[dict[str, torch.Tensor]], pad_token_id: int) -> dict[str, torch.Tensor]:
        max_len = max(x["input_ids"].size(0) for x in batch)
        input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        for i, x in enumerate(batch):
            L = x["input_ids"].size(0)
            input_ids[i, :L] = x["input_ids"]
            attention_mask[i, :L] = x["attention_mask"]
            labels[i, :L] = x["labels"]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def _train_dpo(self, preference_pairs: list[dict[str, Any]], context_embeddings: torch.Tensor):
        """Custom DPO training loop. Trains projector + LoRA; reference model = LoRA disabled."""
        kw = self.dpo_train_kwargs
        epochs: int = kw.get("epochs", 1)
        batch_size: int = kw.get("batch_size", 2)
        grad_accum_steps: int = kw.get("grad_accum_steps", 8)
        lr: float = kw.get("lr", 1e-4)
        beta: float = kw.get("beta", 0.1)
        max_length: int = kw.get("max_length", self.soft_token_model_config.get("llm_max_length", 2048))
        log_every: int = kw.get("log_every", 10)
        seed: int = kw.get("seed", 42)

        model = self._soft_token_model
        assert model is not None
        tokenizer = model.llm_tokenizer

        # Pre-tokenize all pairs.
        print(f"[SoftTokenPersonalizationSystem]   Tokenizing {len(preference_pairs)} pairs...")
        tokenized: list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]] = []
        for p in preference_pairs:
            c = self._tokenize_pair(tokenizer, p["current_messages"], p["chosen"], max_length)
            r = self._tokenize_pair(tokenizer, p["current_messages"], p["rejected"], max_length)
            tokenized.append((c, r))

        # Trainable params: projector + LoRA adapter params.
        trainable = [p for p in model.projector.parameters()]
        trainable += [p for p in model.llm.parameters() if p.requires_grad]
        print(f"[SoftTokenPersonalizationSystem]   Trainable tensors: {len(trainable)} (projector + LoRA)")
        optimizer = torch.optim.AdamW(trainable, lr=lr)

        device = next(model.llm.parameters()).device
        pad_id = tokenizer.pad_token_id

        n = len(preference_pairs)
        rng = torch.Generator().manual_seed(seed)

        global_step = 0
        for epoch in range(epochs):
            perm = torch.randperm(n, generator=rng).tolist()
            running_loss = 0.0
            running_acc = 0.0
            running_count = 0
            optimizer.zero_grad()

            pbar = tqdm(range(0, n, batch_size), desc=f"DPO epoch {epoch}")
            for step, start in enumerate(pbar):
                idx = perm[start:start + batch_size]
                chosen_batch = self._collate([tokenized[i][0] for i in idx], pad_id)
                rejected_batch = self._collate([tokenized[i][1] for i in idx], pad_id)
                ctx = context_embeddings[idx].to(device)

                # Policy log-probs (LoRA enabled)
                model.llm.train()
                model.projector.train()
                pol_chosen_logp = model.forward_with_soft_tokens(
                    ctx, chosen_batch["input_ids"], chosen_batch["attention_mask"],
                    chosen_batch["labels"], use_adapter=True,
                )
                pol_rejected_logp = model.forward_with_soft_tokens(
                    ctx, rejected_batch["input_ids"], rejected_batch["attention_mask"],
                    rejected_batch["labels"], use_adapter=True,
                )

                # Reference log-probs (LoRA disabled, no grad)
                with torch.no_grad():
                    ref_chosen_logp = model.forward_with_soft_tokens(
                        ctx, chosen_batch["input_ids"], chosen_batch["attention_mask"],
                        chosen_batch["labels"], use_adapter=False,
                    )
                    ref_rejected_logp = model.forward_with_soft_tokens(
                        ctx, rejected_batch["input_ids"], rejected_batch["attention_mask"],
                        rejected_batch["labels"], use_adapter=False,
                    )

                pol_logratios = pol_chosen_logp - pol_rejected_logp
                ref_logratios = ref_chosen_logp - ref_rejected_logp
                logits = beta * (pol_logratios - ref_logratios)
                loss = -F.logsigmoid(logits).mean() / grad_accum_steps
                loss.backward()

                with torch.no_grad():
                    acc = (logits > 0).float().mean().item()
                running_loss += loss.item() * grad_accum_steps
                running_acc += acc
                running_count += 1

                if (step + 1) % grad_accum_steps == 0 or start + batch_size >= n:
                    torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                if (step + 1) % log_every == 0:
                    avg_loss = running_loss / running_count
                    avg_acc = running_acc / running_count
                    pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.3f}")
                global_step += 1

            print(
                f"[SoftTokenPersonalizationSystem]   Epoch {epoch} done. "
                f"avg_loss={running_loss / max(1, running_count):.4f}, "
                f"avg_acc={running_acc / max(1, running_count):.3f}"
            )

    # -----------------------------------------------------------------------
    # EVALUATE
    # -----------------------------------------------------------------------

    def evaluate(
        self,
        dataset: PersonalizationDataset,
        load_path: Path,
        gt_user_attributes: dict[str, list[dict[str, str]]] | None = None,
    ) -> list[str]:
        load_path = Path(load_path)
        print("[SoftTokenPersonalizationSystem] 1. Loading trained soft-token model...")
        with open(load_path / "soft_token_model_config.json", "r") as f:
            model_config_dict = json.load(f)
        model_config = SoftTokenLLMConfig(**model_config_dict)
        self._soft_token_model = SoftTokenLLM(model_config)
        self._soft_token_model.load_trained(load_path)
        log_gpu_usage("After soft-token model load (eval)")

        # Encode histories
        print("[SoftTokenPersonalizationSystem] 2. Encoding histories...")
        history_texts = [_format_history(dataset[i].conversation_history) for i in range(len(dataset))]
        context_embeddings = self._soft_token_model.encode_history(
            history_texts, batch_size=self.encoder_batch_size
        )

        # Generate
        print("[SoftTokenPersonalizationSystem] 3. Generating personalized responses...")
        model = self._soft_token_model
        tokenizer = model.llm_tokenizer
        # Left padding for generation.
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        gen_batch_size: int = self.generation_kwargs.pop("batch_size", 4) if "batch_size" in self.generation_kwargs else 4
        outputs: list[str] = []
        for start in tqdm(range(0, len(dataset), gen_batch_size), desc="Generating"):
            idx = list(range(start, min(start + gen_batch_size, len(dataset))))
            # Apply chat template to prompts (no response) and pad.
            prompt_id_lists = [
                _apply_chat_template_ids(
                    tokenizer, dataset[i].current_messages, add_gen_prompt=True,
                )
                for i in idx
            ]
            max_len = max(len(x) for x in prompt_id_lists)
            pad_id = tokenizer.pad_token_id
            input_ids = torch.full((len(idx), max_len), pad_id, dtype=torch.long)
            attention_mask = torch.zeros((len(idx), max_len), dtype=torch.long)
            for j, ids in enumerate(prompt_id_lists):
                L = len(ids)
                input_ids[j, max_len - L:] = torch.tensor(ids, dtype=torch.long)
                attention_mask[j, max_len - L:] = 1
            ctx = context_embeddings[idx]

            gen_ids = model.generate(
                ctx, input_ids, attention_mask, **self.generation_kwargs,
            )
            # With inputs_embeds, HF returns only the newly generated ids.
            decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            outputs.extend(decoded)

        del context_embeddings, model
        self._soft_token_model.unload()
        self._soft_token_model = None
        import gc as _gc
        _gc.collect()
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
        log_gpu_usage("After soft-token model unload (eval)")
        return outputs
