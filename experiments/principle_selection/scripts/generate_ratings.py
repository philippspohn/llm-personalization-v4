import hydra
from omegaconf import DictConfig
from llm_personalization.data.load_ultrachat import load_ultrachat_conversations
from candidate_principles import candidate_principles

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    conversations = load_ultrachat_conversations(
        split="train_sft",
        limit=100,
        seed=42,
    )

    judge_prompts = []

    for conversation in conversations:
        for i in range(len(conversation) - 1):
            prompt = conversation[i]["content"]
            response = conversation[i + 1]["content"]
            for principle in candidate_principles:
                judge_prompts.append((prompt, response, principle))

    print(f"Loaded {len(conversations)} conversations")
    print(conversations[0])

if __name__ == "__main__":
    main()