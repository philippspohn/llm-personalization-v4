import hydra
from omegaconf import DictConfig
from llm_personalization.data.load_ultrachat import load_ultrachat_conversations
from llm_personalization.judge import PrincipleJudge
from hydra.utils import instantiate
import random


TEST_PRINCIPLES = [
    "formal",
    "casual",
    "formality",
    "The response uses formal, stiff, impersonal language",
    "The response uses casual, relaxed, conversational language",
    "The response uses formal, stiff, impersonal language (as opposed to casual, relaxed, conversational language)",
]
TEST_CONVERSATIONS = [
    # Clearly concise
    [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris."},
    ],
    # Clearly verbose
    [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "That's a great question! The capital of France is Paris, which is a beautiful city located in the north-central part of the country along the Seine River. Paris has been the capital of France for many centuries and is known worldwide for its stunning architecture, rich history, world-class museums like the Louvre, iconic landmarks such as the Eiffel Tower, and its vibrant culture. The city is also a major European hub for commerce, fashion, gastronomy, and the arts. I hope that helps!"},
    ],
]

TEST_PRINCIPLES = [
    "concise",
    "verbose",
    "formal",
    "casual",
    "The response is concise and brief",
    "The response is verbose and long-winded",
]

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # conversations = load_ultrachat_conversations(
    #     split="train_sft",
    #     limit=3,
    #     seed=42,
    # )
    
    conversations = TEST_CONVERSATIONS

    rng2 = random.Random(42)

    print("\n\n=== CONVERSATIONS ===\n")
    for i, messages in enumerate(conversations):
        number_of_messages = rng2.randint(1, len(messages) // 2) * 2
        messages = messages[:number_of_messages]
        print(f"--- Conv {i+1} ({len(messages)} messages) ---")
        for msg in messages:
            print(f"  {msg['role'].upper()}: {msg['content']}")
        print()

    judge: PrincipleJudge = instantiate(cfg.judge)
    judge.load()

    rng = random.Random(42)

    # Prepare requests
    judge_prompts = []
    judge_principles = []
    for messages in conversations:
        number_of_messages = rng.randint(1, len(messages) // 2) * 2
        messages = messages[:number_of_messages]
        for principle in TEST_PRINCIPLES:
            judge_prompts.append(messages)
            judge_principles.append(principle)

    scores = judge.judge_principle(
        conversations=judge_prompts,
        principles=judge_principles,
    )

    # Display results
    n_principles = len(TEST_PRINCIPLES)
    n_conversations = len(conversations)
    expected_scores = n_principles * n_conversations
    if len(scores) != expected_scores:
        raise ValueError(
            f"Unexpected number of scores: got {len(scores)}, expected {expected_scores}"
        )

    conv_headers = " ".join(f"{f'Conv{i + 1}':>6}" for i in range(n_conversations))
    print(f"\n{'Principle':<75} {conv_headers}")
    print("-" * (76 + 7 * n_conversations))
    for p_idx, principle in enumerate(TEST_PRINCIPLES):
        scores_for_principle = [
            scores[c_idx * n_principles + p_idx] for c_idx in range(n_conversations)
        ]
        label = principle[:72] + "..." if len(principle) > 75 else principle
        score_cols = " ".join(f"{score:>6.3f}" for score in scores_for_principle)
        print(f"{label:<75} {score_cols}")


if __name__ == "__main__":
    main()