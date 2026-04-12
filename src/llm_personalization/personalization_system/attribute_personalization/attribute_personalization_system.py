from llm_personalization.benchmark.personalization_system import PersonalizationSystem

from llm_personalization.benchmark.attribute_benchmark.attribute_personalization_dataset import AttributePersonalizationDataset
from llm_personalization.benchmark.personalization_judge import PersonalizationJudge
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from llm_personalization.classification_model.text_classification_model import TextClassificationModel
from llm_personalization.llm.llm_helper import LLMHelper
from typing import Literal
import json
import torch
import pandas as pd
from llm_personalization.utils.gpu_monitor import log_gpu_usage

SYSTEM_PROMPT_TEMPLATE = """
Your task is to respond to the user's prompt while strictly adhering to the following response principle:
{direction} {attribute}

You must {direction_instruction}.
"""

def _format_system_prompt(attribute: str, side: str) -> str:
    if side == "follow":
        return SYSTEM_PROMPT_TEMPLATE.format(
            direction="FOLLOW",
            attribute=attribute,
            direction_instruction=f"demonstrate strong {attribute} in your response"
        )
    else:
        return SYSTEM_PROMPT_TEMPLATE.format(
            direction="AVOID",
            attribute=attribute,
            direction_instruction=f"avoid any {attribute} in your response"
        )


class AttributePersonalizationSystem(PersonalizationSystem):
    def __init__(self, text_classification_model_config: dict, llm_helper_config: dict, attributes: list[str], attribute_selection: Literal["abs", "abs_two_sided", "margin"] = "abs", text_classification_model_train_kwargs: dict = {}, predict_batch_size: int | None = None):
        self.text_classification_model_config = text_classification_model_config
        self.llm_helper_config = llm_helper_config
        self.text_classification_model = TextClassificationModel(**self.text_classification_model_config, num_classes=len(attributes)*2) # 2 classes per attribute (avoid/follow)
        self.llm_helper = LLMHelper(**self.llm_helper_config)
        self.attributes = attributes
        self.attribute_selection = attribute_selection
        self.text_classification_model_train_kwargs = text_classification_model_train_kwargs
        self.predict_batch_size = predict_batch_size

    def _format_history(self, history: list[list[dict[str, str]]]) -> str:
        text = ""
        for conversation in history:
            text += f"<conversation>\n"
            for message in conversation:
                text += f"<message role='{message['role']}'>{message['content']}</message>\n"
            text += f"</conversation>\n"
        return text

    
    def _save_attributes(self, path: Path, attributes: list[str]):
        with open(path, "w") as f:
            json.dump(list(attributes), f)

    def _load_attributes(self, path: Path) -> list[str]:
        with open(path, "r") as f:
            return json.load(f)


    def train(self, dataset: AttributePersonalizationDataset, judge: PersonalizationJudge, save_path: Path, gt_user_attributes: dict[str, list[dict[str, str]]] | None = None):
        # 1. Extract labels
        print("[AttributePersonalizationSystem] 1.   Extracting labels...")
        # 1.a. Generate responses for each item and attribute
        print("[AttributePersonalizationSystem] 1.a. Generating responses...")
        generation_prompts = []
        generation_metadata = []
        for i, item in enumerate(dataset):
            for attribute in self.attributes:
                generation_prompts.append([
                    {"role": "system", "content": _format_system_prompt(attribute, "follow")},
                ] + item.current_messages) # TODO: randomize range?
                generation_metadata.append({"user_id": item.user_id, "attribute": attribute, "side": "follow", "current_messages": item.current_messages})
                if self.attribute_selection in ("margin", "abs_two_sided"):
                    generation_prompts.append([
                        {"role": "system", "content": _format_system_prompt(attribute, "avoid")},
                    ] + item.current_messages)
                    generation_metadata.append({"user_id": item.user_id, "attribute": attribute, "side": "avoid", "current_messages": item.current_messages})

        log_gpu_usage("Before LLM load")
        print("[AttributePersonalizationSystem]      Loading LLM helper...")
        self.llm_helper.load()
        log_gpu_usage("After LLM load")
        print("[AttributePersonalizationSystem]    Generating responses...")
        responses = self.llm_helper.generate(generation_prompts)
        print("[AttributePersonalizationSystem]      Responses generated.")
        self.llm_helper.unload()
        log_gpu_usage("After LLM unload")

        # 1.b. Judge responses
        print("[AttributePersonalizationSystem] 1.b. Judging responses...")
        judge_user_ids = [m["user_id"] for m in generation_metadata]
        judge_requests = [
            m["current_messages"] + [{"role": "assistant", "content": response.content}]
            for m, response in zip(generation_metadata, responses)
        ]

        log_gpu_usage("Before judge load")
        print("[AttributePersonalizationSystem]      Loading judge...")
        judge.load()
        log_gpu_usage("After judge load")
        print("[AttributePersonalizationSystem]      Judging responses...")
        raw_judge_scores = judge.judge(judge_user_ids, judge_requests)
        print("[AttributePersonalizationSystem]      Responses judged.")
        judge.unload()
        log_gpu_usage("After judge unload")

        # 1.c. Select labels
        print("[AttributePersonalizationSystem] 1.c. Selecting labels...")
        attr_to_idx = {attr: i for i, attr in enumerate(self.attributes)}
        df = pd.DataFrame([{"user_id": m["user_id"], "attribute": m["attribute"], "side": m["side"]} for m in generation_metadata])
        df["score"] = raw_judge_scores

        # Debug: check for None/NaN scores
        none_count = sum(1 for s in raw_judge_scores if s is None)
        nan_count = df["score"].isna().sum()
        print(f"[AttributePersonalizationSystem]      None scores: {none_count}/{len(raw_judge_scores)}, NaN in df: {nan_count}/{len(df)}")

        # Replace NaN scores with per-attribute mean so they don't get selected
        if nan_count > 0:
            df["score"] = df.groupby("attribute")["score"].transform(lambda s: s.fillna(s.mean()))
            remaining_nan = df["score"].isna().sum()
            if remaining_nan > 0:
                df["score"] = df["score"].fillna(0)
            print(f"[AttributePersonalizationSystem]      After NaN fill: {df['score'].isna().sum()} NaN remaining")

        if self.attribute_selection == "abs":
            # Pick the attribute with the highest absolute score per user, shifted by 5.5 so scores are centered around 0
            df["centered_score"] = df["score"] - 5.5
            best_idx = df.groupby("user_id")["centered_score"].apply(lambda s: s.abs().idxmax())
            best = df.loc[best_idx].copy()
            best["side"] = best["centered_score"].apply(lambda s: "avoid" if s < 0 else "follow")

        elif self.attribute_selection == "abs_two_sided":
            # Generate both follow and avoid probes, then pick the (attribute, side) with the highest raw score per user
            best_idx = df.groupby("user_id")["score"].idxmax()
            best = df.loc[best_idx].copy()
            # side is already set from generation_metadata ("follow" or "avoid")

        elif self.attribute_selection == "margin":
            # Compute margin = follow_score - avoid_score per (user, attribute), pick highest margin
            pivoted = df.pivot_table(index=["user_id", "attribute"], columns="side", values="score").reset_index()
            pivoted["margin"] = pivoted["follow"] - pivoted["avoid"]
            nan_margins = pivoted["margin"].isna().sum()
            print(f"[AttributePersonalizationSystem]      NaN margins: {nan_margins}/{len(pivoted)}")
            if nan_margins > 0:
                pivoted["margin"] = pivoted["margin"].fillna(0)
            best_idx = pivoted.groupby("user_id")["margin"].apply(lambda s: s.abs().idxmax())
            best = pivoted.loc[best_idx].copy()
            best["side"] = best["margin"].apply(lambda m: "follow" if m > 0 else "avoid")

        # Encode label: avoid = attr_idx, follow = attr_idx + len(attributes)
        best["label"] = best.apply(
            lambda row: attr_to_idx[row["attribute"]] + (len(self.attributes) if row["side"] == "follow" else 0), axis=1
        )
        labels_by_user_id = dict(zip(best["user_id"], best["label"]))

        # Debug: label distribution
        label_dist = pd.Series(list(labels_by_user_id.values())).value_counts().sort_index()
        print(f"[AttributePersonalizationSystem]      Label distribution ({len(labels_by_user_id)} users, {len(label_dist)} classes used):")
        for label, count in label_dist.items():
            attr = self.attributes[label % len(self.attributes)]
            side = ['avoid', 'follow'][label // len(self.attributes)]
            print(f"        {label}: {attr} ({side}) = {count}")

        # Debug: print first 3 users
        for user_id in list(labels_by_user_id.keys())[:3]:
            label = labels_by_user_id[user_id]
            user_df = df[df["user_id"] == user_id]
            print(f"User {user_id} scores: {dict(zip(user_df['attribute'], user_df['score']))}")
            print(f"User {user_id} → {self.attributes[label % len(self.attributes)]} ({['avoid', 'follow'][label // len(self.attributes)]})")


        # Debug: judge accuracy (how well does label selection match GT)
        if gt_user_attributes is not None:
            correct = 0
            total = 0
            attr_correct = 0
            for user_id, label in labels_by_user_id.items():
                if user_id in gt_user_attributes:
                    gt_attrs = gt_user_attributes[user_id]
                    # GT is list of {"attribute": ..., "side": ...}
                    for gt in gt_attrs:
                        if gt["attribute"] in attr_to_idx:
                            gt_label = attr_to_idx[gt["attribute"]] + (len(self.attributes) if gt["side"] == "follow" else 0)
                            total += 1
                            if label == gt_label:
                                correct += 1
                            if self.attributes[label % len(self.attributes)] == gt["attribute"]:
                                attr_correct += 1
            if total > 0:
                print(f"[AttributePersonalizationSystem]      JUDGE ACCURACY (label matches GT): {correct}/{total} = {correct/total:.4f}")
                print(f"[AttributePersonalizationSystem]      JUDGE ATTR ACCURACY (attribute only, ignoring side): {attr_correct}/{total} = {attr_correct/total:.4f}")

        # 2. Train text classification model
        print("[AttributePersonalizationSystem] 2.   Training text classification model...")
        print("[AttributePersonalizationSystem] 2.a. Formatting texts...")
        texts = []
        labels = []
        for item in dataset:
            user_id = item.user_id
            label = labels_by_user_id[user_id]
            text = self._format_history(item.conversation_history)
            texts.append(text)
            labels.append(label)

        log_gpu_usage("Before text classification model load")
        print(f"[AttributePersonalizationSystem] 2.b. Loading text classification model...")
        self.text_classification_model.load_untrained()
        print(f"[AttributePersonalizationSystem]      Training text classification model...")
        self.text_classification_model.train(texts, labels, **self.text_classification_model_train_kwargs) # TODO: implement different losses (e.g. regression loss)

        # 3. Save text classification model
        print("[AttributePersonalizationSystem] 3.    Saving text classification model...")
        self.text_classification_model.save_to_file(save_path / "text_classification_model")
        self._save_attributes(save_path / "attributes.json", self.attributes)
        # TODO save config too?

    def evaluate(self, dataset: AttributePersonalizationDataset, load_path: Path, gt_user_attributes: dict[str, list[dict[str, str]]] | None = None) -> list[str]:
        print("[AttributePersonalizationSystem] Evaluating text classification model...")
        print("[AttributePersonalizationSystem] 1. Predicting attributes...")

        attributes = self._load_attributes(load_path / "attributes.json")
        print("[AttributePersonalizationSystem]    Formatting texts...")
        texts = []
        for item in dataset:
            text = self._format_history(item.conversation_history)
            texts.append(text)
        print("[AttributePersonalizationSystem]    Loading text classification model...")
        self.text_classification_model.load_from_file(load_path / "text_classification_model")
        print("[AttributePersonalizationSystem]    Predicting attributes...")
        predictions = self.text_classification_model.predict(texts, batch_size=self.predict_batch_size) # TODO: predict multiple attributes
        print("[AttributePersonalizationSystem]    Attributes predicted.")
        self.text_classification_model.unload()

        # Debug: test accuracy (how well does classifier match GT)
        if gt_user_attributes is not None:
            attr_to_idx = {attr: i for i, attr in enumerate(attributes)}
            correct = 0
            attr_correct = 0
            total = 0
            for item, prediction in zip(dataset, predictions):
                if item.user_id in gt_user_attributes:
                    gt_attrs = gt_user_attributes[item.user_id]
                    for gt in gt_attrs:
                        if gt["attribute"] in attr_to_idx:
                            gt_label = attr_to_idx[gt["attribute"]] + (len(attributes) if gt["side"] == "follow" else 0)
                            total += 1
                            if prediction == gt_label:
                                correct += 1
                            if attributes[prediction % len(attributes)] == gt["attribute"]:
                                attr_correct += 1
            if total > 0:
                print(f"[AttributePersonalizationSystem]    TEST ACCURACY (label matches GT): {correct}/{total} = {correct/total:.4f}")
                print(f"[AttributePersonalizationSystem]    TEST ATTR ACCURACY (attribute only, ignoring side): {attr_correct}/{total} = {attr_correct/total:.4f}")

        print("[AttributePersonalizationSystem] 2. Generating responses...")
        generation_prompts = []
        for item, prediction in zip(dataset, predictions):
            attribute = attributes[prediction % len(self.attributes)]
            side = ['avoid', 'follow'][prediction // len(self.attributes)]
            generation_prompts.append([
                {"role": "system", "content": _format_system_prompt(attribute, side)},
            ] + item.current_messages)

        print("[AttributePersonalizationSystem]    Loading LLM helper...")
        self.llm_helper.load()
        print("[AttributePersonalizationSystem]    Generating responses...")
        responses = self.llm_helper.generate(generation_prompts)
        print("[AttributePersonalizationSystem]    Responses generated.")
        self.llm_helper.unload()
        return [response.content for response in responses]