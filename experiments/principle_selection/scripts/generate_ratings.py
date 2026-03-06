import hydra
from omegaconf import DictConfig
from llm_personalization.data.load_ultrachat import load_ultrachat_conversations
from candidate_principles import candidate_principles
from llm_personalization.judge import PrincipleJudge
from hydra.utils import instantiate, get_original_cwd
import random
import numpy as np
from pathlib import Path

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    conversations = load_ultrachat_conversations(
        split="train_sft",
        limit=cfg.limit,
        seed=42,
    )

    judge: PrincipleJudge = instantiate(cfg.judge)
    judge.load()

    rng = random.Random(42)

    judge_requests_prompts = []
    judge_requests_principles = []

    for messages in conversations:
        if len(messages) < 2:
            raise ValueError(f"Conversation has less than 2 messages: {messages}")
        number_of_messages = rng.randint(1, len(messages) // 2) * 2
        messages = messages[:number_of_messages]
        for principle in candidate_principles:
            judge_requests_prompts.append(messages)
            judge_requests_principles.append(principle)

    scores = judge.judge_principle(
        conversations=judge_requests_prompts,
        principles=judge_requests_principles,
    )

    num_conversations = len(conversations)
    num_principles = len(candidate_principles)
    matrix = np.array(scores).reshape(num_conversations, num_principles)
    out_path = Path(get_original_cwd()) / cfg.output_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, matrix)


if __name__ == "__main__":
    main()