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
from llm_personalization.utils.gpu_monitor import log_gpu_usage

# SYSTEM_PROMPT_TEMPLATE = """
# Adopt the following communication style attributes in all your responses:

# {attributes}

# Let these attributes naturally shape your tone, word choice, and structure.
# """
SYSTEM_PROMPT_TEMPLATE = """
Adopt the following communication style attribute in your responses

Attribute: {attribute}

Let this attribute naturally shape your tone, word choice, and structure.
"""


class AttributePersonalizationSystem(PersonalizationSystem):
    def __init__(self, text_classification_model_config: dict, llm_helper_config: dict, attributes: list[str], attribute_selection: Literal["abs", "margin"] = "abs", text_classification_model_train_kwargs: dict = {}, predict_batch_size: int | None = None):
        self.text_classification_model_config = text_classification_model_config
        self.llm_helper_config = llm_helper_config
        self.text_classification_model = TextClassificationModel(**self.text_classification_model_config, num_classes=len(attributes)*2) # 2 classes per attribute (avoid/follow)
        self.llm_helper = LLMHelper(**self.llm_helper_config)
        self.attributes = attributes
        self.attribute_selection = attribute_selection
        self.text_classification_model_train_kwargs = text_classification_model_train_kwargs
        self.predict_batch_size = predict_batch_size
        if attribute_selection == "margin":
            print(f"[AttributePersonalizationSystem] Attribute selection set to: margin. Attribute list must be antonym pairs. E.g. {self.attributes[0]} - {self.attributes[1]}")

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

    def train(self, dataset: AttributePersonalizationDataset, judge: PersonalizationJudge, save_path: Path):
        # 1. Extract labels
        print("[AttributePersonalizationSystem] 1.   Extracting labels...")
        # 1.a. Generate responses for each item and attribute
        print("[AttributePersonalizationSystem] 1.a. Generating responses...")
        generation_prompts = []
        for item in dataset:
            for attribute in self.attributes:
                generation_prompts.append([
                    {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(attribute=attribute)},
                ] + item.current_messages) # TODO: randomize range?

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
        response_idx = 0
        judge_requests = []
        judge_user_ids = []
        for item in dataset:
            for attribute in self.attributes:
                response = responses[response_idx]
                response_idx += 1
                messages = item.current_messages + [{"role": "assistant", "content": response}] # TODO: randomize range?
                judge_requests.append(messages)
                judge_user_ids.append(item.user_id)

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
        label_idx = 0
        labels_by_user_id = {}
        for i, item in enumerate(dataset):
            user_scores = []
            for attribute in self.attributes:
                score = raw_judge_scores[label_idx]
                label_idx += 1
                user_scores.append(score)
            argmax_score = torch.argmax(torch.tensor(user_scores).abs()).item()
            side = "avoid" if user_scores[argmax_score] < 0 else "follow"
            label = argmax_score
            if side == "follow":
                label = label + len(self.attributes)
            labels_by_user_id[item.user_id] = label
            if i < 3:
                avg_msg_length = sum(len(msg) for msg in item.conversation_history) / len(item.conversation_history)
                print(f"User {item.user_id} history: {len(item.conversation_history)} messages, avg length: {avg_msg_length}")
                print(f"User {item.user_id} current messages: {item.current_messages}")
                print(f"Available attributes: {self.attributes}")
                print(f"User {item.user_id} scores: {user_scores}") # TODO: should be zero-centered?
                print(f"User {item.user_id} label: {argmax_score}")
                print(f"User {item.user_id} label: {self.attributes[label % len(self.attributes)]}")
                print(f"User {item.user_id} side: {['avoid', 'follow'][label // len(self.attributes)]}")
                assert side == ['avoid', 'follow'][label // len(self.attributes)]

        # user_id_to_scores = {}
        # if self.attribute_selection == "abs":
        #     for judge_user_id, score in zip(judge_user_ids, raw_judge_scores):
        #         if judge_user_id not in user_id_to_scores:
        #             user_id_to_scores[judge_user_id] = []
        #         user_id_to_scores[judge_user_id].append(score)
        # elif self.attribute_selection == "margin":
        #     # TODO: margin mode needs rework — current pairing logic assumes exactly 2 attributes
        #     # and breaks when users have more. Revisit the score pairing strategy.
        #     raise NotImplementedError("Margin attribute selection is not yet fully implemented")
        #     for i in range(0, len(judge_user_ids), 2):
        #         user_id = judge_user_ids[i]
        #         score_positive = raw_judge_scores[i]
        #         score_negative = raw_judge_scores[i + 1]
        #         score_margin = score_positive - score_negative
        #         if user_id not in user_id_to_scores:
        #             user_id_to_scores[user_id] = []
        #         user_id_to_scores[user_id].append(score_margin)

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

    def evaluate(self, dataset: AttributePersonalizationDataset, load_path: Path) -> list[str]:
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

        print("[AttributePersonalizationSystem] 2. Generating responses...")
        generation_prompts = []
        for item, prediction in zip(dataset, predictions):
            attribute = attributes[prediction % len(self.attributes)]
            side = ['avoid', 'follow'][prediction // len(self.attributes)]
            if side == "avoid":
                attribute = f"not {attribute}"
            generation_prompts.append([
                {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(attribute=attribute)},
            ] + item.current_messages)

        print("[AttributePersonalizationSystem]    Loading LLM helper...")
        self.llm_helper.load()
        print("[AttributePersonalizationSystem]    Generating responses...")
        responses = self.llm_helper.generate(generation_prompts)
        print("[AttributePersonalizationSystem]    Responses generated.")
        self.llm_helper.unload()
        return [response.content for response in responses]