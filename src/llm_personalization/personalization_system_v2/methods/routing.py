"""Trained-routing personalization method.

Trains a text classifier on (user_history_text -> response_attribute_label),
where labels are derived from the cached candidate ratings (not from fresh
judge calls). At test time we predict a (response_attribute, side) pair per
user and inject it into the same Nemotron-style system prompt that was used
to generate the cached candidates.

Two label modes (selectable via config):

  * argmax     - hard cross-entropy on the (attr, side) with the highest
                 cached rating against the user's GT.
  * regression - soft cross-entropy on the softmax of the full 2K rating
                 vector against the user's GT axis.

The classifier itself is the existing TextClassificationModel
(ModernBERT-base by default), reused unchanged.
"""

from __future__ import annotations

from llm_personalization.classification_model.text_classification_model import (
    TextClassificationModel,
)
from llm_personalization.llm.llm_helper import LLMHelper

from ..dataset import PersonalizationExample
from ..label_selection import LabelMode, decode_class, derive_labels, format_history_text
from ..method import MethodInput, PersonalizationMethod
from ..prompts import format_system_prompt


class TrainedRouting(PersonalizationMethod):
    needs_gt_at_test = False

    def __init__(
        self,
        llm: LLMHelper,
        attributes: list[str],
        text_classification_model: dict,         # kwargs for TextClassificationModel
        label_mode: LabelMode = "argmax",
        train_kwargs: dict | None = None,
        predict_batch_size: int = 16,
        max_chars_per_msg: int = 1024,
    ):
        if label_mode not in ("argmax", "regression", "margin", "abs_centered"):
            raise ValueError(f"Unsupported label_mode={label_mode!r}")
        self.llm = llm
        self.attributes = list(attributes)
        self.label_mode = label_mode
        # 2 sides per attribute -> 2K classes
        self.classifier = TextClassificationModel(
            num_classes=len(self.attributes) * 2,
            **text_classification_model,
        )
        self.train_kwargs = train_kwargs or {}
        self.predict_batch_size = predict_batch_size
        self.max_chars_per_msg = max_chars_per_msg

    # ------------------------------------------------------------------ fit
    def fit(self, train: list[PersonalizationExample]) -> None:
        kept, labels = derive_labels(train, self.attributes, self.label_mode)
        if not kept:
            raise RuntimeError(
                "No usable training examples after label derivation "
                "(no examples with both ratings and GT?)"
            )
        print(
            f"[TrainedRouting] Derived {len(labels)} labels (mode={self.label_mode!r}, "
            f"dropped {len(train) - len(kept)})",
            flush=True,
        )

        if self.label_mode == "argmax":
            from collections import Counter
            dist = Counter(labels)
            print(
                f"[TrainedRouting] Label distribution ({len(dist)}/{2*len(self.attributes)} "
                f"classes used):",
                flush=True,
            )
            for cls, count in sorted(dist.items()):
                attr_i, side = decode_class(cls, len(self.attributes))
                print(f"  {cls:>3}: {self.attributes[attr_i]} ({side}) = {count}", flush=True)

        texts = [format_history_text(ex, self.max_chars_per_msg) for ex in kept]

        print(f"[TrainedRouting] Loading classifier (untrained)...", flush=True)
        self.classifier.load_untrained()
        print(f"[TrainedRouting] Training classifier on {len(texts)} examples...", flush=True)
        self.classifier.train(texts, labels, **self.train_kwargs)

    # ------------------------------------------------------------------ generate
    def generate(self, batch: list[MethodInput]) -> list[str]:
        # 1) Predict labels for the test users.
        texts = [
            format_history_text(
                # Wrap MethodInput in a dummy PersonalizationExample for formatting.
                PersonalizationExample(
                    user_id=item.user_id,
                    split="test",
                    gt_user_attributes=[],
                    history=item.history,
                    prompt=item.prompt,
                    candidates=[],
                ),
                self.max_chars_per_msg,
            )
            for item in batch
        ]
        print(f"[TrainedRouting] Predicting attributes for {len(texts)} users...", flush=True)
        predictions = self.classifier.predict(texts, batch_size=self.predict_batch_size)
        self.classifier.unload()

        # 2) Build personalized prompts and run a single batched LLM call.
        prompts = []
        for item, pred in zip(batch, predictions):
            attr_idx, side = decode_class(pred, len(self.attributes))
            sys_prompt = format_system_prompt(self.attributes[attr_idx], side)
            prompts.append([{"role": "system", "content": sys_prompt}, *item.prompt])

        # Diagnostic: prediction distribution
        from collections import Counter
        dist = Counter(predictions)
        print(f"[TrainedRouting] Prediction distribution at test time:", flush=True)
        for cls, count in sorted(dist.items()):
            attr_i, side = decode_class(cls, len(self.attributes))
            print(f"  {cls:>3}: {self.attributes[attr_i]} ({side}) = {count}", flush=True)

        self.llm.load()
        try:
            responses = self.llm.generate(prompts)
        finally:
            self.llm.unload()
        return [r.content for r in responses]
