"""Derive per-user training labels from the cached candidate ratings.

For each train user we have a 2K-vector of "response-attribute scores":
the judge's rating of each generated candidate against the user's GT
(attribute, side). Concretely, for a candidate generated under
(gen_attr, gen_side) and a user with GT = (gt_attr, gt_side):

    score[gen_attr, gen_side] = candidate.ratings[(gt_attr, gt_side)]

(The "avoid" side of the GT is the cached `derived: True` flip of the
"follow" rating, i.e. `11 - direct_score`. So we just read the cached
value directly without re-flipping.)

This module turns that 2K-vector into a label suitable for a text
classifier head. The class index encodes (gen_attr, gen_side) as:

    class_idx = attr_idx + (len(attributes) if side == "follow" else 0)

(matches the convention used by v1's AttributePersonalizationSystem.)

Label modes:
  - "argmax":       hard label = argmax(score_vector); cross-entropy loss
  - "regression":   soft label = softmax(score_vector); soft cross-entropy
  - "margin":       per-attribute margin (follow - avoid); pick attribute with
                    largest |margin|, side from sign(margin)
  - "abs_centered": score - 5.5; pick (attr, side) with largest |centered|;
                    side = "avoid" if centered<0 else "follow", attribute is
                    fixed by the candidate's own (gen_attr).

Each function returns either an int (hard label), or a list[float] of length
2K summing to 1 (soft label).
"""

from __future__ import annotations

import math
from typing import Literal

from .dataset import PersonalizationExample, Side


LabelMode = Literal["argmax", "regression", "margin", "abs_centered"]


def encode_class(attr_idx: int, side: Side, num_attributes: int) -> int:
    return attr_idx + (num_attributes if side == "follow" else 0)


def decode_class(class_idx: int, num_attributes: int) -> tuple[int, Side]:
    return class_idx % num_attributes, ("follow" if class_idx >= num_attributes else "avoid")


def _score_vector_against_gt(
    example: PersonalizationExample, attributes: list[str]
) -> dict[tuple[str, Side], float] | None:
    """Per-(gen_attr, gen_side) score against the user's *first* GT axis.

    Returns None if the user has no GT or no ratings, so callers can skip.
    The first GT pair is used (matches the 1-attr datasets we currently have);
    extending to multi-attr GT is a TODO -- e.g. average across GT axes.
    """
    if not example.gt_user_attributes or not example.has_ratings:
        return None
    gt_attr = example.gt_user_attributes[0]["attribute"]
    gt_side: Side = example.gt_user_attributes[0]["side"]

    scores: dict[tuple[str, Side], float] = {}
    for c in example.candidates:
        rating = next(
            (r for r in c.ratings if r.target_attribute == gt_attr and r.target_side == gt_side),
            None,
        )
        if rating is None:
            return None  # incomplete ratings; skip user
        scores[(c.attribute, c.side)] = rating.score
    return scores


def derive_labels(
    examples: list[PersonalizationExample],
    attributes: list[str],
    mode: LabelMode,
    *,
    softmax_temperature: float = 1.0,
) -> tuple[list[PersonalizationExample], list[int | list[float]]]:
    """Filter examples that have everything needed and derive labels.

    Returns (kept_examples, labels) of equal length. Examples that lack GT,
    candidates, or ratings are silently dropped.
    """
    attr_to_idx = {a: i for i, a in enumerate(attributes)}
    n_attrs = len(attributes)

    kept: list[PersonalizationExample] = []
    labels: list[int | list[float]] = []

    for ex in examples:
        scores = _score_vector_against_gt(ex, attributes)
        if scores is None:
            continue

        if mode == "argmax":
            (best_attr, best_side), _ = max(scores.items(), key=lambda kv: kv[1])
            cls = encode_class(attr_to_idx[best_attr], best_side, n_attrs)
            labels.append(cls)

        elif mode == "regression":
            score_vec = [0.0] * (2 * n_attrs)
            for (attr, side), s in scores.items():
                score_vec[encode_class(attr_to_idx[attr], side, n_attrs)] = s
            t = softmax_temperature
            m = max(score_vec)
            exps = [math.exp((s - m) / t) for s in score_vec]
            z = sum(exps)
            soft = [e / z for e in exps]
            labels.append(soft)

        elif mode == "abs_centered":
            (best_attr, best_side), best_score = max(
                scores.items(), key=lambda kv: abs(kv[1] - 5.5)
            )
            side: Side = "avoid" if best_score < 5.5 else "follow"
            cls = encode_class(attr_to_idx[best_attr], side, n_attrs)
            labels.append(cls)

        elif mode == "margin":
            best_attr_idx = -1
            best_margin = -math.inf
            best_side: Side = "follow"
            for attr in attributes:
                f = scores.get((attr, "follow"))
                a = scores.get((attr, "avoid"))
                if f is None or a is None:
                    continue
                margin = f - a
                if abs(margin) > abs(best_margin):
                    best_margin = margin
                    best_attr_idx = attr_to_idx[attr]
                    best_side = "follow" if margin > 0 else "avoid"
            if best_attr_idx == -1:
                continue
            labels.append(encode_class(best_attr_idx, best_side, n_attrs))

        else:
            raise ValueError(f"Unknown label mode: {mode!r}")

        kept.append(ex)

    return kept, labels


def format_history_text(example: PersonalizationExample, max_chars_per_msg: int = 1024) -> str:
    """Flatten an example's history into a single text the classifier can ingest.

    Mirrors `AttributePersonalizationSystem._format_history` but flattens to
    one string and includes the held-out prompt so the classifier sees the
    full pre-generation context. Per-message content is truncated to keep
    the input within ModernBERT's 8k token budget on average.
    """
    parts: list[str] = []
    for conv in example.history:
        parts.append("<conversation>")
        for msg in conv:
            content = msg["content"]
            if len(content) > max_chars_per_msg:
                content = content[:max_chars_per_msg] + "..."
            parts.append(f"<message role='{msg['role']}'>{content}</message>")
        parts.append("</conversation>")
    parts.append("<current>")
    for msg in example.prompt:
        content = msg["content"]
        if len(content) > max_chars_per_msg:
            content = content[:max_chars_per_msg] + "..."
        parts.append(f"<message role='{msg['role']}'>{content}</message>")
    parts.append("</current>")
    return "\n".join(parts)
