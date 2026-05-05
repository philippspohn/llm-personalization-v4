"""User-embedding personalization (DPO-style with SimPO loss).

Architecture:
  history (text) -> frozen embedder -> linear projector -> soft prompt token
                                                          v
  prompt tokens -> embed -> [soft_token | prompt_embeds | response_embeds] -> LLM (LoRA)

Training:
  Per train user, pick best/worst cached responses (chosen/rejected) by
  weighted user-simulator score against the world's GT targets. Train
  projector + LoRA jointly with the SimPO loss

      L = -log sigma(
              beta * (logp(chosen)/|chosen|  -  logp(rejected)/|rejected|)
              - gamma
          )

  No reference model.

Eval:
  Encode test history -> soft token -> HF model.generate(inputs_embeds=...).
  Slow vs vLLM but workable for ~1k users.
"""
from __future__ import annotations

import gc
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_personalization.utils.gpu_monitor import log_gpu_usage

from ..cache import CachedDataset, CachedUser
from ..prompts import format_history
from .base import PersonalizationSystemV3
from .rag_simple import _pick_best_worst


# ----------------------------- preference data -----------------------------


def _build_pairs(
    dataset: CachedDataset,
    user_id_to_weighted_attrs: dict[str, list[dict]],
) -> list[dict]:
    """Returns list of {history_text, current_messages, chosen, rejected} per user."""
    pairs = []
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
        chosen, rejected = picked
        pairs.append({
            "user_id": user.user_id,
            "history_text": format_history(user.history),
            "current_messages": user.current_messages,
            "chosen": chosen,
            "rejected": rejected,
        })
    if skipped:
        print(f"[UserEmbedding] skipped {skipped} users without usable cached scores")
    return pairs


# ----------------------------- model -----------------------------


class Projector(nn.Module):
    """Small MLP from embedder dim -> LLM hidden size, producing one soft token."""
    def __init__(self, in_dim: int, hidden_size: int, mlp_hidden: int | None = None):
        super().__init__()
        h = mlp_hidden or hidden_size
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.GELU(),
            nn.Linear(h, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, in_dim) -> (B, hidden)
        return self.net(x)


def _embedder_load(model_name: str, max_seq_length: int | None):
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(model_name, device=device, trust_remote_code=True)
    if max_seq_length is not None:
        embedder.max_seq_length = max_seq_length
    return embedder


def _embedder_release(embedder) -> None:
    try:
        embedder.to("cpu")
    except Exception:
        pass
    del embedder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _encode_histories(embedder, histories: list[str], batch_size: int) -> np.ndarray:
    return embedder.encode(
        histories,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,  # raw vectors -> projector decides scale
        show_progress_bar=True,
    ).astype(np.float32)


def _encode_histories_grad(embedder, histories: list[str], device) -> torch.Tensor:
    """Forward through the sentence-transformer with grads enabled. Returns a
    differentiable (B, D) tensor — used when train_embedder=True so the
    embedder can be jointly fine-tuned with the projector + LoRA."""
    features = embedder.tokenize(histories)
    features = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in features.items()}
    out = embedder(features)
    return out["sentence_embedding"]  # (B, D), grad flows through


# ----------------------------- training utils -----------------------------


def _tokenize_prompt(tokenizer, current_messages: list[dict]) -> list[int]:
    """Apply chat template up to the assistant turn (no response yet)."""
    return tokenizer.apply_chat_template(
        current_messages, add_generation_prompt=True, tokenize=True
    )


def _tokenize_response(tokenizer, response: str) -> list[int]:
    """Tokenize the assistant response (no chat-template wrapping; we splice
    it after the chat-template-applied prompt)."""
    ids = tokenizer.encode(response, add_special_tokens=False)
    # Append EOS so the model learns to stop.
    if tokenizer.eos_token_id is not None:
        ids = ids + [tokenizer.eos_token_id]
    return ids


def _logp_response(
    model,
    embed_layer: nn.Embedding,
    soft_tokens: torch.Tensor,        # (B, hidden)
    prompt_ids: list[list[int]],      # B prompts
    response_ids: list[list[int]],    # B responses
    pad_token_id: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (sum_logp, num_response_tokens) per item in batch.

    Builds inputs_embeds = [soft, prompt_embeds, response_embeds], computes
    log P(response_token_t | everything before it). Pads on the right with
    pad_token_id (masked out in loss).
    """
    B = len(prompt_ids)
    seq_lens = [1 + len(p) + len(r) for p, r in zip(prompt_ids, response_ids)]
    max_len = max(seq_lens)

    # Build padded token-id matrix for the prompt+response part (ignoring soft slot)
    # We allocate length (max_len), with [pad,...] at soft slot replaced via embeds.
    token_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
    response_mask = torch.zeros((B, max_len), dtype=torch.bool, device=device)

    for i, (p, r) in enumerate(zip(prompt_ids, response_ids)):
        # positions: [0]=soft, [1..1+len(p))=prompt, [1+len(p)..1+len(p)+len(r))=response
        token_ids[i, 1 : 1 + len(p)] = torch.tensor(p, device=device)
        token_ids[i, 1 + len(p) : 1 + len(p) + len(r)] = torch.tensor(r, device=device)
        attention_mask[i, : 1 + len(p) + len(r)] = 1
        response_mask[i, 1 + len(p) : 1 + len(p) + len(r)] = True

    # Inject soft token at position 0 via concat (avoids in-place write into
    # a view of the embedding weight, which is a leaf w/ requires_grad).
    word_embeds_rest = embed_layer(token_ids[:, 1:]).to(dtype)        # (B, L-1, H)
    soft = soft_tokens.to(dtype).unsqueeze(1)                          # (B, 1, H)
    word_embeds = torch.cat([soft, word_embeds_rest], dim=1)           # (B, L, H)

    out = model(
        inputs_embeds=word_embeds,
        attention_mask=attention_mask,
        use_cache=False,
    )
    logits = out.logits  # (B, L, V)
    # log P(token at position t) = log_softmax(logits[t-1])[token_t]
    log_probs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    target = token_ids[:, 1:]
    gathered = log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1)  # (B, L-1)
    loss_mask = response_mask[:, 1:]                                  # (B, L-1)

    sum_logp = (gathered * loss_mask).sum(dim=1)                      # (B,)
    n_tokens = loss_mask.sum(dim=1).clamp(min=1)                      # (B,)
    return sum_logp, n_tokens


def _logp_response_no_soft(
    model,
    embed_layer: nn.Embedding,
    prompt_ids: list[list[int]],
    response_ids: list[list[int]],
    pad_token_id: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Like `_logp_response` but with no soft-token slot. Used to score under
    the reference (base) model: same prompt + response, but the policy's soft
    prefix is dropped. Always called inside `torch.no_grad()`."""
    B = len(prompt_ids)
    seq_lens = [len(p) + len(r) for p, r in zip(prompt_ids, response_ids)]
    max_len = max(seq_lens)

    token_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
    response_mask = torch.zeros((B, max_len), dtype=torch.bool, device=device)
    for i, (p, r) in enumerate(zip(prompt_ids, response_ids)):
        token_ids[i, : len(p)] = torch.tensor(p, device=device)
        token_ids[i, len(p) : len(p) + len(r)] = torch.tensor(r, device=device)
        attention_mask[i, : len(p) + len(r)] = 1
        response_mask[i, len(p) : len(p) + len(r)] = True

    word_embeds = embed_layer(token_ids).to(dtype)
    out = model(inputs_embeds=word_embeds, attention_mask=attention_mask, use_cache=False)
    logits = out.logits
    log_probs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    target = token_ids[:, 1:]
    gathered = log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1)
    loss_mask = response_mask[:, 1:]
    sum_logp = (gathered * loss_mask).sum(dim=1)
    n_tokens = loss_mask.sum(dim=1).clamp(min=1)
    return sum_logp, n_tokens


def _simpo_loss(
    sum_logp_chosen, n_chosen,
    sum_logp_rejected, n_rejected,
    beta: float, gamma: float,
) -> torch.Tensor:
    avg_chosen = sum_logp_chosen / n_chosen
    avg_rejected = sum_logp_rejected / n_rejected
    margin = beta * (avg_chosen - avg_rejected) - gamma
    loss = -F.logsigmoid(margin)
    return loss.mean(), avg_chosen.detach().mean(), avg_rejected.detach().mean()


def _dpo_loss(
    sum_logp_chosen, sum_logp_rejected,
    ref_sum_logp_chosen, ref_sum_logp_rejected,
    beta: float,
) -> torch.Tensor:
    """Standard DPO with the unchanged base LLM (LoRA-off, no soft token) as
    the reference. Uses sum-of-log-probs (not length-normalized — that's
    SimPO's job)."""
    pol_diff = sum_logp_chosen - sum_logp_rejected
    ref_diff = ref_sum_logp_chosen - ref_sum_logp_rejected
    margin = beta * (pol_diff - ref_diff)
    loss = -F.logsigmoid(margin)
    avg_c = (sum_logp_chosen - ref_sum_logp_chosen).detach().mean()
    avg_r = (sum_logp_rejected - ref_sum_logp_rejected).detach().mean()
    return loss.mean(), avg_c, avg_r


# ----------------------------- main class -----------------------------


class UserEmbeddingSystem(PersonalizationSystemV3):
    def __init__(
        self,
        gen_model: str,
        embedder_model: str,
        attributes: list[str],
        # projector
        projector_mlp_hidden: int | None = None,
        # lora
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: list[str] | None = None,
        # training
        batch_size: int = 1,
        grad_accum_steps: int = 8,
        epochs: int = 1,
        lr: float = 5e-5,
        max_prompt_tokens: int = 1024,
        max_response_tokens: int = 1024,
        # loss
        loss_type: str = "simpo",   # "simpo" | "dpo"
        beta: float = 2.0,           # SimPO β (paper default 2.0); DPO uses dpo_beta
        gamma: float = 1.0,          # SimPO margin γ
        dpo_beta: float = 0.1,       # DPO β (paper default 0.1)
        seed: int = 42,
        gradient_checkpointing: bool = True,
        # inference
        infer_batch_size: int = 4,
        max_new_tokens: int = 1024,
        infer_dtype: str = "bfloat16",
        # embedder
        embedder_batch_size: int = 32,
        embedder_max_seq_length: int | None = 512,
        train_embedder: bool = False,
        embedder_lr: float | None = None,  # default = lr; small embedder may want a different LR
    ):
        self.gen_model = gen_model
        self.embedder_model = embedder_model
        self.attributes = list(attributes)

        self.projector_mlp_hidden = projector_mlp_hidden
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        # Cast away Hydra's ListConfig so peft's JSON serializer can save it.
        self.lora_target_modules = list(lora_target_modules) if lora_target_modules else ["q_proj", "k_proj", "v_proj", "o_proj"]

        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.epochs = epochs
        self.lr = lr
        self.max_prompt_tokens = max_prompt_tokens
        self.max_response_tokens = max_response_tokens
        if loss_type not in ("simpo", "dpo"):
            raise ValueError(f"loss_type must be 'simpo' or 'dpo', got {loss_type!r}")
        self.loss_type = loss_type
        self.beta = beta
        self.gamma = gamma
        self.dpo_beta = dpo_beta
        self.seed = seed
        self.gradient_checkpointing = gradient_checkpointing

        self.infer_batch_size = infer_batch_size
        self.max_new_tokens = max_new_tokens
        self.infer_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[infer_dtype]

        self.embedder_batch_size = embedder_batch_size
        self.embedder_max_seq_length = embedder_max_seq_length
        self.train_embedder = train_embedder
        self.embedder_lr = embedder_lr if embedder_lr is not None else lr

    # --------------- helpers ---------------

    def _load_llm(self, dtype: torch.dtype):
        tokenizer = AutoTokenizer.from_pretrained(self.gen_model, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            self.gen_model, torch_dtype=dtype, trust_remote_code=True,
        )
        if torch.cuda.is_available():
            model = model.cuda()
        return tokenizer, model

    def _make_lora(self, model):
        cfg = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, cfg)
        return model

    # --------------- train ---------------

    def train(
        self,
        dataset: CachedDataset,
        user_id_to_weighted_attrs: dict[str, list[dict]],
        save_path: Path,
        val_dataset: CachedDataset | None = None,
        val_user_id_to_weighted_attrs: dict[str, list[dict]] | None = None,
    ) -> None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        # 1. Build train + val pairs
        print("[UserEmbedding] 1. Building preference pairs...")
        train_pairs = _build_pairs(dataset, user_id_to_weighted_attrs)
        val_pairs = (
            _build_pairs(val_dataset, val_user_id_to_weighted_attrs)
            if (val_dataset is not None and val_user_id_to_weighted_attrs is not None) else []
        )
        print(f"[UserEmbedding]    train pairs: {len(train_pairs)}, val pairs: {len(val_pairs)}")

        # 2. Encode histories
        # If train_embedder=False (default): pre-encode upfront, free embedder.
        # If train_embedder=True: keep embedder loaded, encode per batch with grad.
        embedder = _embedder_load(self.embedder_model, self.embedder_max_seq_length)
        log_gpu_usage("After embedder load")
        if not self.train_embedder:
            print("[UserEmbedding] 2. Encoding histories with frozen embedder (one-shot)...")
            train_emb = _encode_histories(embedder, [p["history_text"] for p in train_pairs], self.embedder_batch_size)
            val_emb = (
                _encode_histories(embedder, [p["history_text"] for p in val_pairs], self.embedder_batch_size)
                if val_pairs else np.zeros((0, train_emb.shape[1]), dtype=np.float32)
            )
            emb_dim = train_emb.shape[1]
            _embedder_release(embedder)
            embedder = None
            log_gpu_usage("After embedder unload")
        else:
            print("[UserEmbedding] 2. train_embedder=True — embedder stays loaded, encoded per batch with grad.")
            # Probe one history to learn the output dim
            with torch.no_grad():
                probe = _encode_histories_grad(embedder, [train_pairs[0]["history_text"]], device="cuda")
            emb_dim = int(probe.shape[1])
            train_emb = None  # not pre-computed
            val_emb = None
            for p in embedder.parameters():
                p.requires_grad = True
            log_gpu_usage("After embedder configure")

        # 3. Load LLM + LoRA + projector
        print(f"[UserEmbedding] 3. Loading LLM ({self.gen_model}) + LoRA + projector...")
        log_gpu_usage("Before LLM load")
        tokenizer, model = self._load_llm(self.infer_dtype)
        # Freeze base model; LoRA wrapper will mark its own params trainable.
        for p in model.parameters():
            p.requires_grad = False
        model = self._make_lora(model)
        model.print_trainable_parameters()
        if self.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

        hidden_size = model.config.hidden_size
        # Keep projector in fp32 for numerical stability; cast its output to
        # the LLM's dtype just before injecting into inputs_embeds.
        projector = Projector(emb_dim, hidden_size, self.projector_mlp_hidden).cuda()
        for p in projector.parameters():
            p.requires_grad = True
        log_gpu_usage("After LLM+LoRA+projector load")

        # 4. Pre-tokenize prompts + responses
        print("[UserEmbedding] 4. Tokenizing prompts and responses...")
        def _prep(pairs):
            prompts, chosens, rejecteds = [], [], []
            for p in pairs:
                pid = _tokenize_prompt(tokenizer, p["current_messages"])[: self.max_prompt_tokens]
                cid = _tokenize_response(tokenizer, p["chosen"])[: self.max_response_tokens]
                rid = _tokenize_response(tokenizer, p["rejected"])[: self.max_response_tokens]
                prompts.append(pid); chosens.append(cid); rejecteds.append(rid)
            return prompts, chosens, rejecteds
        tr_p, tr_c, tr_r = _prep(train_pairs)
        va_p, va_c, va_r = _prep(val_pairs)

        # 5. Training loop (SimPO or DPO)
        embed_layer = model.get_input_embeddings()
        # Two parameter groups: main (projector + LoRA) and embedder (different LR).
        main_params = list(projector.parameters()) + [p for p in model.parameters() if p.requires_grad]
        param_groups = [{"params": main_params, "lr": self.lr}]
        if self.train_embedder:
            param_groups.append({"params": list(embedder.parameters()), "lr": self.embedder_lr})
        optimizer = torch.optim.AdamW(param_groups)
        all_params = main_params + (list(embedder.parameters()) if self.train_embedder else [])
        device = next(model.parameters()).device

        def _ref_logps(pids, ys):
            """Score response tokens under the reference (LoRA-off, no soft).
            Always under no_grad; flips the adapter on/off internally."""
            with torch.no_grad():
                # PEFT: disable adapter for the reference forward
                ctx = model.disable_adapter() if hasattr(model, "disable_adapter") else None
                if ctx is not None:
                    with ctx:
                        sl, _ = _logp_response_no_soft(
                            model, embed_layer, pids, ys, tokenizer.pad_token_id, device, self.infer_dtype
                        )
                else:
                    sl, _ = _logp_response_no_soft(
                        model, embed_layer, pids, ys, tokenizer.pad_token_id, device, self.infer_dtype
                    )
            return sl.detach()

        def _step_loss(idx, split):
            """split: "train" or "val" — selects token arrays + history texts."""
            if split == "train":
                pids, cids, rids = [tr_p[i] for i in idx], [tr_c[i] for i in idx], [tr_r[i] for i in idx]
                hist_texts = [train_pairs[i]["history_text"] for i in idx]
                emb_arr = train_emb
            else:
                pids, cids, rids = [va_p[i] for i in idx], [va_c[i] for i in idx], [va_r[i] for i in idx]
                hist_texts = [val_pairs[i]["history_text"] for i in idx]
                emb_arr = val_emb
            if self.train_embedder:
                emb = _encode_histories_grad(embedder, hist_texts, device).to(torch.float32)
            else:
                emb = torch.from_numpy(emb_arr[idx]).to(device, dtype=torch.float32)
            soft = projector(emb)
            sl_c, n_c = _logp_response(model, embed_layer, soft, pids, cids, tokenizer.pad_token_id, device, self.infer_dtype)
            sl_r, n_r = _logp_response(model, embed_layer, soft, pids, rids, tokenizer.pad_token_id, device, self.infer_dtype)
            if self.loss_type == "simpo":
                loss, avg_c, avg_r = _simpo_loss(sl_c, n_c, sl_r, n_r, self.beta, self.gamma)
            else:  # dpo
                ref_sl_c = _ref_logps(pids, cids)
                ref_sl_r = _ref_logps(pids, rids)
                loss, avg_c, avg_r = _dpo_loss(sl_c, sl_r, ref_sl_c, ref_sl_r, self.dpo_beta)
            # pref_acc uses length-normalized score regardless of loss type
            with torch.no_grad():
                pref_acc = ((sl_c / n_c) > (sl_r / n_r)).float().sum().item()
            return loss, avg_c, avg_r, pref_acc

        def _eval_val_loss():
            if not val_pairs:
                return None
            model.eval(); projector.eval()
            if self.train_embedder: embedder.eval()
            with torch.no_grad():
                total, n = 0.0, 0
                for s in range(0, len(val_pairs), self.batch_size):
                    e = min(s + self.batch_size, len(val_pairs))
                    idx = list(range(s, e))
                    loss, _, _, _ = _step_loss(idx, "val")
                    total += loss.item() * (e - s); n += (e - s)
            model.train(); projector.train()
            if self.train_embedder: embedder.train()
            return total / max(1, n)

        loss_descr = (f"SimPO (β={self.beta}, γ={self.gamma})" if self.loss_type == "simpo"
                      else f"DPO (β={self.dpo_beta})")
        print(f"[UserEmbedding] 5. Training loss={loss_descr}: epochs={self.epochs}, "
              f"batch={self.batch_size}, grad_accum={self.grad_accum_steps}")
        model.train(); projector.train()
        if self.train_embedder: embedder.train()
        rng = np.random.default_rng(self.seed)
        for epoch in range(self.epochs):
            order = rng.permutation(len(train_pairs))
            running, running_acc, running_n = 0.0, 0.0, 0
            optimizer.zero_grad()
            for step, idx_start in enumerate(tqdm(range(0, len(order), self.batch_size), desc=f"epoch {epoch}")):
                idx = order[idx_start : idx_start + self.batch_size].tolist()
                loss, avg_c, avg_r, pref_acc = _step_loss(idx, "train")
                (loss / self.grad_accum_steps).backward()
                running += loss.item() * len(idx); running_n += len(idx)
                running_acc += pref_acc
                if (step + 1) % self.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                    optimizer.step(); optimizer.zero_grad()
                if (step + 1) % 50 == 0:
                    print(f"    step {step+1}: loss={running/running_n:.4f} pref_acc={running_acc/running_n:.3f} "
                          f"avg_c={avg_c.item():.2f} avg_r={avg_r.item():.2f}")
            optimizer.step(); optimizer.zero_grad()
            val_loss = _eval_val_loss()
            tag = f" val_loss={val_loss:.4f}" if val_loss is not None else ""
            print(f"  Epoch {epoch}: train_loss={running/running_n:.4f} pref_acc={running_acc/running_n:.3f}{tag}")

        # 6. Save
        print(f"[UserEmbedding] 6. Saving LoRA + projector + emb_dim to {save_path}")
        model.save_pretrained(save_path / "lora")
        tokenizer.save_pretrained(save_path / "lora")
        torch.save(projector.state_dict(), save_path / "projector.pt")
        if self.train_embedder:
            embedder.save(str(save_path / "embedder"))
            _embedder_release(embedder)
            embedder = None
        with open(save_path / "config.json", "w") as f:
            json.dump({
                "gen_model": self.gen_model,
                "embedder_model": self.embedder_model,
                "train_embedder": self.train_embedder,
                "emb_dim": int(emb_dim),
                "hidden_size": int(hidden_size),
                "projector_mlp_hidden": self.projector_mlp_hidden,
                "lora_r": self.lora_r,
                "loss_type": self.loss_type,
                "beta": self.beta,
                "gamma": self.gamma,
                "dpo_beta": self.dpo_beta,
            }, f, indent=2)

        # Free training-time memory before benchmark moves on
        del model, projector, embed_layer, optimizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --------------- evaluate ---------------

    def evaluate(
        self,
        dataset: CachedDataset,
        load_path: Path,
        user_id_to_weighted_attrs: dict[str, list[dict]] | None = None,
    ) -> list[str]:
        load_path = Path(load_path)
        with open(load_path / "config.json") as f:
            cfg = json.load(f)
        emb_dim = cfg["emb_dim"]
        hidden_size = cfg["hidden_size"]

        # 1. Encode test histories — load fine-tuned embedder if it was saved.
        print("[UserEmbedding] 1. Encoding test histories...")
        log_gpu_usage("Before embedder load")
        embedder_path = load_path / "embedder"
        if cfg.get("train_embedder", False) and embedder_path.exists():
            print(f"[UserEmbedding]    using fine-tuned embedder at {embedder_path}")
            embedder = _embedder_load(str(embedder_path), self.embedder_max_seq_length)
        else:
            embedder = _embedder_load(self.embedder_model, self.embedder_max_seq_length)
        test_emb = _encode_histories(
            embedder, [format_history(u.history) for u in dataset], self.embedder_batch_size
        )
        _embedder_release(embedder)
        log_gpu_usage("After embedder unload")

        # 2. Load LLM + LoRA + projector
        print("[UserEmbedding] 2. Loading LLM + LoRA adapter + projector...")
        log_gpu_usage("Before LLM load")
        tokenizer, base = self._load_llm(self.infer_dtype)
        model = PeftModel.from_pretrained(base, str(load_path / "lora"))
        model.eval()
        projector = Projector(emb_dim, hidden_size, self.projector_mlp_hidden).cuda()
        projector.load_state_dict(torch.load(load_path / "projector.pt", map_location="cuda"))
        projector.eval()
        embed_layer = model.get_input_embeddings()
        device = next(model.parameters()).device
        log_gpu_usage("After LLM+LoRA+projector load")

        # 3. Generate (HF generate with inputs_embeds; left-pad batches)
        print("[UserEmbedding] 3. Generating (HF model.generate with inputs_embeds)...")
        responses: list[str] = [""] * len(dataset)
        prompts_ids = [_tokenize_prompt(tokenizer, u.current_messages)[: self.max_prompt_tokens] for u in dataset]
        all_indices = list(range(len(dataset)))

        for s in tqdm(range(0, len(all_indices), self.infer_batch_size), desc="generate"):
            idx = all_indices[s : s + self.infer_batch_size]
            B = len(idx)
            emb = torch.from_numpy(test_emb[idx]).to(device, dtype=torch.float32)
            with torch.no_grad():
                soft = projector(emb).to(self.infer_dtype)  # (B, H)
            batch_prompts = [prompts_ids[i] for i in idx]
            max_p = max(len(p) for p in batch_prompts)
            # Layout per row (length = 1 + max_p):
            #   [pad_left ... pad_left, soft, prompt_token_0, ..., prompt_token_{n-1}]
            # Right-aligned so generation continues after the prompt.
            max_len = 1 + max_p
            input_ids = torch.full((B, max_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
            attention_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
            soft_pos = torch.empty(B, dtype=torch.long, device=device)
            for j, p in enumerate(batch_prompts):
                offset = max_len - len(p)              # prompt starts here
                soft_pos[j] = offset - 1               # soft sits one slot earlier
                input_ids[j, offset:] = torch.tensor(p, device=device)
                attention_mask[j, offset - 1 :] = 1    # soft + real tokens
            # Build embeds via index-aware combination (no in-place on leaf views)
            base_embeds = embed_layer(input_ids).to(self.infer_dtype)            # (B, L, H)
            soft_full = torch.zeros_like(base_embeds)                            # (B, L, H)
            soft_full[torch.arange(B, device=device), soft_pos] = soft           # place soft per row
            mask = torch.zeros((B, max_len, 1), dtype=base_embeds.dtype, device=device)
            mask[torch.arange(B, device=device), soft_pos] = 1.0
            word_embeds = base_embeds * (1.0 - mask) + soft_full * mask
            with torch.no_grad():
                out_ids = model.generate(
                    inputs_embeds=word_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            # When inputs_embeds is used, generate returns only the new tokens.
            for j, ids in zip(idx, out_ids):
                responses[j] = tokenizer.decode(ids, skip_special_tokens=True)

        # Cleanup
        del model, base, projector, embed_layer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log_gpu_usage("After eval unload")
        return responses
