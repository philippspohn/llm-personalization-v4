import os
import hydra
from omegaconf import DictConfig
import numpy as np
from llm_personalization.data.load_ultrachat import load_ultrachat_conversations


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    conversations = load_ultrachat_conversations(
        split="train_sft",
        limit=cfg.limit // len(cfg.pairs) * len(cfg.pairs),
        seed=42,
    )

    pairs = cfg.pairs
    num_pairs = len(pairs)
    conv_per_pair = len(conversations) // num_pairs

    conversations_to_judge = []
    attributes_to_judge = []
    for i, conversation in enumerate(conversations):
        pair = pairs[i % num_pairs]
        conversations_to_judge.extend([conversation, conversation])
        attributes_to_judge.extend(pair)

    output_dir = os.path.join(hydra.utils.get_original_cwd(), "data", "compare_judges")
    os.makedirs(output_dir, exist_ok=True)

    for judge_key, judge_config in cfg.judge_configs.items():
        judge = hydra.utils.instantiate(judge_config)
        judge.load()

        scores = judge.judge_response_attribute(conversations_to_judge, attributes_to_judge)

        judge.unload()

        # (2*N,) -> (N, 2) -> (conv_per_pair, num_pairs, 2) -> (num_pairs, 2, conv_per_pair)
        scores = (
            np.array(scores)
            .reshape(len(conversations), 2)
            .reshape(conv_per_pair, num_pairs, 2)
            .transpose(1, 2, 0)
        )

        np.savez(
            os.path.join(output_dir, f"{judge_key}.npz"),
            scores=scores,           # (num_pairs, 2, conv_per_pair)
            pairs=np.array(list(pairs)),  # (num_pairs, 2)
        )
        print(f"Saved {judge_key} -> {output_dir}/{judge_key}.npz")


if __name__ == "__main__":
    main()