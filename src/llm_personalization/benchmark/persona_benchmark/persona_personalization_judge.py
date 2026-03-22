from llm_personalization.judge.judge import PersonaJudge
from typing import Any
from llm_personalization.benchmark.personalization_judge import PersonalizationJudge


class PersonaPersonalizationJudge(PersonalizationJudge):
    def __init__(self, persona_judge: PersonaJudge, user_id_to_formatted_persona: dict[str, str]):
        self.persona_judge = persona_judge
        self.user_id_to_formatted_persona = user_id_to_formatted_persona
        self.reset_train_statistics()

    def load(self):
        self.persona_judge.load()

    def unload(self):
        self.persona_judge.unload()

    def get_train_statistics(self) -> dict[str, Any]:
        return self.train_statistics

    def reset_train_statistics(self):
        self.train_statistics = {
            "num_judge_requests": 0,
            "requested_ids": [],
        }

    def update_user_id_mapping(self, user_id_to_formatted_persona: dict[str, str]):
        self.user_id_to_formatted_persona = user_id_to_formatted_persona

    def judge(self, user_ids: list[str], conversations: list[list[dict[str, str]]]) -> list[float]:
        judge_personas: list[str] = []
        judge_requests: list[list[dict[str, str]]] = []

        for user_id, conversation in zip(user_ids, conversations):
            if user_id not in self.user_id_to_formatted_persona:
                raise ValueError(f"Invalid user ID: {user_id}")
            judge_personas.append(self.user_id_to_formatted_persona[user_id])
            judge_requests.append(conversation)

        scores = self.persona_judge.judge_response_persona(judge_requests, judge_personas)

        # Replace None scores with neutral value
        final_scores = []
        for score in scores:
            final_scores.append(score if score is not None else 5.5)

        self.train_statistics["num_judge_requests"] += len(user_ids)
        return final_scores
