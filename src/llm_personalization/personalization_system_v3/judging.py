from __future__ import annotations

from llm_personalization.judge.judge import AttributeJudge


class WeightedAttributeJudge:
    """Eval-time judge that scores each (response) against a user's GT
    weighted target attributes and aggregates as

        sum(w_i * s_i) / sum(|w_i|)

    where s_i is `11 - raw` for `side=='avoid'` (matches v1's flip).

    Delegates to `AttributeJudge.judge_response_attribute(...)`, so all of the
    underlying ParsedRatingJudge behaviour is preserved (thinking-mode handling,
    retries, JUDGE_SYSTEM_PROMPT*, etc.). Issues a single batched call across
    all (user, target) pairs for vLLM efficiency.
    """

    def __init__(
        self,
        attribute_judge: AttributeJudge,
        user_id_to_weighted_attrs: dict[str, list[dict]],
    ):
        self.attribute_judge = attribute_judge
        self.user_id_to_weighted_attrs = user_id_to_weighted_attrs

    def load(self) -> None:
        self.attribute_judge.load()

    def unload(self) -> None:
        self.attribute_judge.unload()

    def judge(
        self,
        user_ids: list[str],
        conversations: list[list[dict[str, str]]],
    ) -> list[float]:
        judge_attributes: list[str] = []
        judge_requests: list[list[dict[str, str]]] = []

        for user_id, conversation in zip(user_ids, conversations):
            if user_id not in self.user_id_to_weighted_attrs:
                raise ValueError(f"Invalid user ID: {user_id}")
            for target in self.user_id_to_weighted_attrs[user_id]:
                judge_attributes.append(target["attribute"])
                judge_requests.append(conversation)

        raw_scores = self.attribute_judge.judge_response_attribute(judge_requests, judge_attributes)

        idx = 0
        out: list[float] = []
        for user_id in user_ids:
            num, denom = 0.0, 0.0
            for target in self.user_id_to_weighted_attrs[user_id]:
                raw = raw_scores[idx]
                idx += 1
                if raw is None:
                    continue
                s = (11.0 - raw) if target["side"] == "avoid" else raw
                w = float(target["weight"])
                num += w * s
                denom += abs(w)
            out.append(num / denom if denom > 0 else float("nan"))
        return out
