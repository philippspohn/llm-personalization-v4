"""
Soft-token LLM wrapper.

Pipeline:
    conversation_history (text)
        -> frozen text encoder (e.g. BAAI/bge-m3) -> pooled embedding (enc_dim)
        -> trainable Linear(enc_dim -> num_soft_tokens * hidden_size) -> soft tokens (N, H)
        -> prepended to the LLM's inputs_embeds
        -> LLM (with trainable LoRA adapter) produces the response

Only two things are trainable:
    1. The linear projector (encoder -> soft tokens)
    2. The LoRA adapter on the LLM

The base LLM weights and the encoder weights are frozen.
"""
from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass
class SoftTokenLLMConfig:
    llm_model: str
    encoder_model: str = "BAAI/bge-m3"
    num_soft_tokens: int = 8
    encoder_max_length: int = 512
    llm_max_length: int = 2048
    llm_dtype: str = "bfloat16"
    encoder_dtype: str = "float32"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None  # None -> peft default for the arch
    device_map: str | None = "auto"  # "auto" shards across visible GPUs; None = single device
    gradient_checkpointing: bool = True


def _get_hidden_size(config) -> int:
    """Handle nested configs (e.g. Gemma3/4, PaliGemma) that put hidden_size in text_config."""
    if hasattr(config, "hidden_size") and config.hidden_size is not None:
        return config.hidden_size
    for attr in ("text_config", "language_model_config", "decoder_config"):
        sub = getattr(config, attr, None)
        if sub is not None and getattr(sub, "hidden_size", None) is not None:
            return sub.hidden_size
    raise AttributeError(f"Could not find hidden_size on config {type(config).__name__}")


def _mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
    summed = (token_embeddings * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class SoftTokenLLM(nn.Module):
    """
    Frozen encoder + trainable Linear projector + LLM with LoRA adapter.

    Forward expects already-encoded context embeddings (to avoid re-encoding the
    same conversation history for chosen/rejected in DPO). Use `encode_history`
    to produce them.
    """

    def __init__(self, config: SoftTokenLLMConfig):
        super().__init__()
        self.config = config
        self.encoder: AutoModel | None = None
        self.encoder_tokenizer: AutoTokenizer | None = None
        self.llm: AutoModelForCausalLM | None = None
        self.llm_tokenizer: AutoTokenizer | None = None
        self.projector: nn.Linear | None = None
        self._hidden_size: int | None = None
        self._encoder_dim: int | None = None

    # ----- load / save ------------------------------------------------------

    def load_base(self, with_lora: bool = True) -> None:
        """Load encoder + LLM from scratch, attach fresh LoRA adapter & projector."""
        cfg = self.config
        encoder_dtype = DTYPE_MAP[cfg.encoder_dtype]
        llm_dtype = DTYPE_MAP[cfg.llm_dtype]

        print(f"[SoftTokenLLM] Loading encoder {cfg.encoder_model}...")
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(cfg.encoder_model, trust_remote_code=True)
        self.encoder = AutoModel.from_pretrained(cfg.encoder_model, trust_remote_code=True, torch_dtype=encoder_dtype)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        self._encoder_dim = self.encoder.config.hidden_size

        print(f"[SoftTokenLLM] Loading LLM {cfg.llm_model}...")
        self.llm_tokenizer = AutoTokenizer.from_pretrained(cfg.llm_model, trust_remote_code=True)
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        llm_load_kwargs: dict[str, Any] = dict(trust_remote_code=True, torch_dtype=llm_dtype)
        if cfg.device_map is not None:
            llm_load_kwargs["device_map"] = cfg.device_map
        llm = AutoModelForCausalLM.from_pretrained(cfg.llm_model, **llm_load_kwargs)
        for p in llm.parameters():
            p.requires_grad = False
        self._hidden_size = _get_hidden_size(llm.config)

        if cfg.gradient_checkpointing:
            # Needed so grads flow through base-model activations to LoRA params.
            llm.gradient_checkpointing_enable()
            if hasattr(llm, "enable_input_require_grads"):
                llm.enable_input_require_grads()

        if with_lora:
            lora_kwargs: dict[str, Any] = dict(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            if cfg.lora_target_modules is not None:
                lora_kwargs["target_modules"] = list(cfg.lora_target_modules)
            lora_config = LoraConfig(**lora_kwargs)
            llm = get_peft_model(llm, lora_config)
            llm.print_trainable_parameters()
        self.llm = llm

        self.projector = nn.Linear(self._encoder_dim, cfg.num_soft_tokens * self._hidden_size, bias=True)
        nn.init.normal_(self.projector.weight, std=1e-3)
        nn.init.zeros_(self.projector.bias)
        self.projector = self.projector.to(torch.float32)

        self._to_device()

    def _to_device(self) -> None:
        """Move non-sharded pieces (encoder, projector) to CUDA.

        The LLM is either already placed by `device_map="auto"` (sharded across
        GPUs) or is still on CPU; in the latter case we move it to cuda:0.
        """
        if not torch.cuda.is_available():
            return
        self.encoder = self.encoder.to("cuda")
        if self.config.device_map is None:
            # Not sharded — put the whole LLM on a single device.
            self.llm = self.llm.to("cuda")
        # Place projector on the same device as the LLM's input-embedding layer so
        # its output can be concatenated without an extra copy.
        emb_layer = self._locate_input_embeddings()
        proj_device = emb_layer.weight.device if emb_layer is not None else torch.device("cuda:0")
        self.projector = self.projector.to(proj_device)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        # LoRA adapter
        self.llm.save_pretrained(path / "lora_adapter")
        # Projector
        torch.save(
            {
                "state_dict": self.projector.state_dict(),
                "encoder_dim": self._encoder_dim,
                "hidden_size": self._hidden_size,
            },
            path / "projector.pt",
        )
        # Tokenizers
        self.llm_tokenizer.save_pretrained(path / "llm_tokenizer")
        self.encoder_tokenizer.save_pretrained(path / "encoder_tokenizer")
        print(f"[SoftTokenLLM] Saved to {path}")

    def load_trained(self, path: Path) -> None:
        """Load encoder + base LLM, then attach saved LoRA adapter and projector."""
        cfg = self.config
        encoder_dtype = DTYPE_MAP[cfg.encoder_dtype]
        llm_dtype = DTYPE_MAP[cfg.llm_dtype]
        path = Path(path)

        print(f"[SoftTokenLLM] Loading encoder {cfg.encoder_model}...")
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(path / "encoder_tokenizer", trust_remote_code=True)
        self.encoder = AutoModel.from_pretrained(cfg.encoder_model, trust_remote_code=True, torch_dtype=encoder_dtype)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        self._encoder_dim = self.encoder.config.hidden_size

        print(f"[SoftTokenLLM] Loading LLM {cfg.llm_model}...")
        self.llm_tokenizer = AutoTokenizer.from_pretrained(path / "llm_tokenizer", trust_remote_code=True)
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        llm_load_kwargs: dict[str, Any] = dict(trust_remote_code=True, torch_dtype=llm_dtype)
        if cfg.device_map is not None:
            llm_load_kwargs["device_map"] = cfg.device_map
        base_llm = AutoModelForCausalLM.from_pretrained(cfg.llm_model, **llm_load_kwargs)
        for p in base_llm.parameters():
            p.requires_grad = False
        self._hidden_size = _get_hidden_size(base_llm.config)
        self.llm = PeftModel.from_pretrained(base_llm, path / "lora_adapter")

        proj_state = torch.load(path / "projector.pt", map_location="cpu")
        self.projector = nn.Linear(proj_state["encoder_dim"], cfg.num_soft_tokens * proj_state["hidden_size"], bias=True)
        self.projector.load_state_dict(proj_state["state_dict"])
        self.projector = self.projector.to(torch.float32)

        self._to_device()
        self.eval()

    def unload(self) -> None:
        # Models loaded with device_map="auto" are wrapped in accelerate dispatch
        # hooks that hold refs to parameter tensors. Naively setting attributes to
        # None doesn't release those refs -> GPU memory stays pinned. We have to:
        #   (1) strip accelerate hooks from every submodule
        #   (2) move the model to CPU (ensures GPU bytes are freed even if any
        #       stray Python ref survives)
        #   (3) drop our references, gc, and empty the per-device caches.
        try:
            from accelerate.hooks import remove_hook_from_module
        except ImportError:
            remove_hook_from_module = None

        def _nuke(model):
            if model is None:
                return
            base = model
            for attr in ("base_model", "model"):
                sub = getattr(base, attr, None)
                if sub is not None and isinstance(sub, nn.Module):
                    base = sub
            if remove_hook_from_module is not None:
                try:
                    remove_hook_from_module(model, recurse=True)
                except Exception:
                    pass
                try:
                    remove_hook_from_module(base, recurse=True)
                except Exception:
                    pass
            try:
                model.to("cpu")
            except Exception:
                pass

        _nuke(getattr(self, "llm", None))
        _nuke(getattr(self, "encoder", None))

        for attr in ("encoder", "llm", "projector", "encoder_tokenizer", "llm_tokenizer"):
            if hasattr(self, attr) and getattr(self, attr) is not None:
                setattr(self, attr, None)
        gc.collect()
        if torch.cuda.is_available():
            # empty_cache() only frees the current device's allocator; loop over all GPUs
            # so vLLM (which inspects free memory per-device) sees released memory.
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    torch.cuda.synchronize()

    # ----- encoding / projection -------------------------------------------

    @torch.no_grad()
    def encode_history(self, history_texts: list[str], batch_size: int = 16) -> torch.Tensor:
        """Encode list of conversation-history strings into (N, encoder_dim) fp32 tensor."""
        assert self.encoder is not None, "Call load_base/load_trained first"
        device = next(self.encoder.parameters()).device
        out: list[torch.Tensor] = []
        for i in range(0, len(history_texts), batch_size):
            batch = history_texts[i:i + batch_size]
            inputs = self.encoder_tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.config.encoder_max_length,
                return_tensors="pt",
            ).to(device)
            outputs = self.encoder(**inputs)
            pooled = _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            out.append(pooled.float().cpu())
        return torch.cat(out, dim=0)

    def project_to_soft_tokens(self, context_embeddings: torch.Tensor) -> torch.Tensor:
        """(B, encoder_dim) fp32 -> (B, num_soft_tokens, hidden_size) in LLM dtype."""
        device = next(self.projector.parameters()).device
        x = context_embeddings.to(device=device, dtype=torch.float32)
        proj = self.projector(x)  # (B, N*H)
        proj = proj.view(x.size(0), self.config.num_soft_tokens, self._hidden_size)
        emb_layer = self._locate_input_embeddings()
        llm_dtype = emb_layer.weight.dtype if emb_layer is not None else next(self.llm.parameters()).dtype
        emb_device = emb_layer.weight.device if emb_layer is not None else device
        return proj.to(device=emb_device, dtype=llm_dtype)

    # ----- forward ----------------------------------------------------------

    def _locate_input_embeddings(self) -> nn.Module | None:
        """Return the input-embedding module of the LLM, walking nested wrappers."""
        if self.llm is None:
            return None
        candidates = [self.llm]
        base = getattr(self.llm, "base_model", None)
        if base is not None:
            candidates.append(base)
            for attr in ("model", "language_model", "text_model"):
                sub = getattr(base, attr, None)
                if sub is not None:
                    candidates.append(sub)
        for c in candidates:
            if hasattr(c, "get_input_embeddings"):
                try:
                    emb = c.get_input_embeddings()
                    if emb is not None:
                        return emb
                except (AttributeError, NotImplementedError):
                    continue
        return None

    def _needs_mm_token_type_ids(self) -> bool:
        """Detect architectures that require mm_token_type_ids (e.g. Gemma 4)."""
        cls_name = type(self.llm).__name__.lower()
        if "gemma4" in cls_name:
            return True
        # Also walk into peft/base wrappers by checking inner class names.
        base = getattr(self.llm, "base_model", None)
        if base is not None and "gemma4" in type(base).__name__.lower():
            return True
        inner = getattr(base, "model", None) if base is not None else None
        if inner is not None and "gemma4" in type(inner).__name__.lower():
            return True
        return False

    def _embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Works for PeftModel, plain CausalLM, and nested architectures (Gemma 3/4, etc.)
        emb_layer = None
        if hasattr(self.llm, "get_input_embeddings"):
            try:
                emb_layer = self.llm.get_input_embeddings()
            except (AttributeError, NotImplementedError):
                emb_layer = None
        if emb_layer is None and hasattr(self.llm, "base_model"):
            try:
                emb_layer = self.llm.base_model.get_input_embeddings()
            except (AttributeError, NotImplementedError):
                emb_layer = None
        if emb_layer is None:
            # Fall back: walk into known nested sub-models.
            base = getattr(self.llm, "base_model", self.llm)
            for attr in ("model", "language_model", "text_model"):
                sub = getattr(base, attr, None)
                if sub is not None and hasattr(sub, "get_input_embeddings"):
                    emb_layer = sub.get_input_embeddings()
                    if emb_layer is not None:
                        break
        if emb_layer is None:
            raise AttributeError("Could not locate input embeddings layer on the LLM")
        return emb_layer(input_ids)

    def forward_with_soft_tokens(
        self,
        context_embeddings: torch.Tensor,  # (B, enc_dim) fp32
        input_ids: torch.Tensor,           # (B, T) includes prompt + response tokens
        attention_mask: torch.Tensor,      # (B, T)
        labels: torch.Tensor | None = None,  # (B, T) with -100 on positions to ignore
        use_adapter: bool = True,
    ) -> torch.Tensor:
        """
        Returns per-sequence sum of log-probabilities of response tokens (where
        labels != -100), shape (B,). Handles the soft-token prefix internally
        by concatenating soft-token embeddings to input embeddings.
        """
        emb_layer = self._locate_input_embeddings()
        device = emb_layer.weight.device if emb_layer is not None else next(self.llm.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)

        soft = self.project_to_soft_tokens(context_embeddings)  # (B, N, H)
        N = soft.size(1)

        token_emb = self._embed_tokens(input_ids)  # (B, T, H)
        inputs_embeds = torch.cat([soft, token_emb], dim=1)  # (B, N+T, H)
        # With gradient checkpointing, the LLM re-runs modules during backward;
        # inputs_embeds must carry grad for the backward path to reach our projector.
        if self.config.gradient_checkpointing and torch.is_grad_enabled():
            inputs_embeds.requires_grad_(True)

        prefix_mask = torch.ones(attention_mask.size(0), N, device=device, dtype=attention_mask.dtype)
        full_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # Some multimodal architectures (e.g. Gemma 4) require mm_token_type_ids
        # during training. For pure-text training we pass all-zeros (text tokens).
        extra_kwargs = {}
        if self._needs_mm_token_type_ids():
            extra_kwargs["mm_token_type_ids"] = torch.zeros(
                full_attention_mask.shape, dtype=torch.long, device=device,
            )

        # Run LLM. For reference-model forward we disable the LoRA adapter.
        if isinstance(self.llm, PeftModel) and not use_adapter:
            with self.llm.disable_adapter():
                out = self.llm(
                    inputs_embeds=inputs_embeds, attention_mask=full_attention_mask, **extra_kwargs,
                )
        else:
            out = self.llm(
                inputs_embeds=inputs_embeds, attention_mask=full_attention_mask, **extra_kwargs,
            )

        logits = out.logits  # (B, N+T, V)

        if labels is None:
            return logits

        # Compute sum log-prob of response tokens.
        # Align logits at positions that predict tokens T[1:]. Soft-token positions
        # contribute no label (they are prefix). We only care about response tokens,
        # which are already marked in `labels` with non -100.
        #
        # Standard shift: logit at position t predicts token at position t+1.
        # Build shifted labels over the full concatenated sequence:
        #   full_labels = [-100]*N concatenated with labels  (B, N+T)
        # Then shift: use logits[:, :-1, :] to predict full_labels[:, 1:].
        # With device_map sharding, logits may live on a different GPU than device.
        logits_device = logits.device
        labels = labels.to(logits_device)
        B, T = input_ids.shape
        full_labels = torch.cat(
            [torch.full((B, N), -100, device=logits_device, dtype=labels.dtype), labels],
            dim=1,
        )  # (B, N+T)
        shift_logits = logits[:, :-1, :].contiguous()  # (B, N+T-1, V)
        shift_labels = full_labels[:, 1:].contiguous()  # (B, N+T-1)

        log_probs = torch.log_softmax(shift_logits.float(), dim=-1)
        # Gather log-probs at target token positions, ignoring -100.
        mask = (shift_labels != -100)
        safe_labels = shift_labels.masked_fill(~mask, 0)
        gathered = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)  # (B, N+T-1)
        gathered = gathered * mask.float()
        per_seq_logp = gathered.sum(dim=-1)  # (B,)
        return per_seq_logp

    # ----- generation -------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        context_embeddings: torch.Tensor,  # (B, enc_dim) fp32
        input_ids: torch.Tensor,           # (B, T) prompt only (left-padded)
        attention_mask: torch.Tensor,
        **generate_kwargs: Any,
    ) -> torch.Tensor:
        """Greedy/sampled generation with the soft-token prefix. Returns (B, T_gen) new tokens only."""
        self.eval()
        emb_layer = self._locate_input_embeddings()
        device = emb_layer.weight.device if emb_layer is not None else next(self.llm.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        soft = self.project_to_soft_tokens(context_embeddings)  # (B, N, H)
        N = soft.size(1)

        token_emb = self._embed_tokens(input_ids)  # (B, T, H)
        inputs_embeds = torch.cat([soft, token_emb], dim=1)
        prefix_mask = torch.ones(attention_mask.size(0), N, device=device, dtype=attention_mask.dtype)
        full_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # Note: when we pass inputs_embeds, HF's generate returns only generated token ids
        # (not the prefix) as of recent versions. We pass pad_token_id explicitly.
        pad_id = self.llm_tokenizer.pad_token_id
        eos_id = self.llm_tokenizer.eos_token_id

        extra_kwargs = {}
        if self._needs_mm_token_type_ids():
            extra_kwargs["mm_token_type_ids"] = torch.zeros(
                full_attention_mask.shape, dtype=torch.long, device=device,
            )

        gen_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
            **extra_kwargs,
            **generate_kwargs,
        )
        return gen_ids
