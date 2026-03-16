from llm_personalization.judge.judge import AttributeJudge
from typing import Any
from llm_personalization.benchmark.personalization_judge import PersonalizationJudge

class PersonalizationAttributeJudge(PersonalizationJudge):
    def __init__(self, attribute_judge: AttributeJudge, user_id_to_response_style_attributes: dict[str, list[str]]):
        self.attribute_judge = attribute_judge
        self.user_id_to_response_style_attributes = user_id_to_response_style_attributes
        self.reset_train_statistics()

    def load(self):
        self.attribute_judge.load()

    def unload(self):
        self.attribute_judge.unload()

    def get_train_statistics(self) -> dict[str, Any]:
        return self.train_statistics

    def reset_train_statistics(self):
        """
        Should not be called by personalization system.
        """
        self.train_statistics = {
            "num_judge_requests": 0,
            "requested_ids": [],
        }

    def update_user_id_mapping(self, user_id_to_response_style_attributes: dict[str, list[str]]):
        """
        Should not be called by personalization system.
        """
        self.user_id_to_response_style_attributes = user_id_to_response_style_attributes

    def judge(self, user_ids: list[str], conversations: list[list[dict[str, str]]]) -> list[float]:
        judge_attributes: list[dict[str, str]] = []
        judge_requests: list[list[dict[str, str]]] = []

        for user_id, conversation in zip(user_ids, conversations):
            if user_id not in self.user_id_to_response_style_attributes:
                raise ValueError(f"Invalid user ID: {user_id}")
            for attribute in self.user_id_to_response_style_attributes[user_id]:
                judge_attributes.append(attribute["attribute"])
                judge_requests.append(conversation)
        scores = self.attribute_judge.judge_response_attribute(judge_requests, judge_attributes)

        score_idx = 0
        final_scores = []
        for user_id in user_ids:
            user_scores = []
            for attribute in self.user_id_to_response_style_attributes[user_id]:
                score = scores[score_idx]
                if attribute["side"] == "avoid":
                    score = -score
                score_idx += 1
                user_scores.append(score)
            final_scores.append(sum(user_scores) / len(user_scores))

        self.train_statistics["num_judge_requests"] += len(user_ids)
        return final_scores