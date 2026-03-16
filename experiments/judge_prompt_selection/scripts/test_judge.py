from llm_personalization.judge import AttributeJudge
from hydra.utils import instantiate
import hydra
from omegaconf import DictConfig
import pandas as pd
import numpy as np


@hydra.main(version_base=None, config_path="../configs", config_name="test_judge")
def main(cfg: DictConfig):
    df = pd.read_csv(cfg.input_path)
    df = df[df["side"] == "follow"]

    attributes = list(cfg.attributes)
    prompt_ids = sorted(df["prompt_id"].unique())

    # Assign a random ground truth attribute per prompt
    rng = np.random.default_rng(seed=42)
    gt_attrs = {pid: rng.choice(attributes) for pid in prompt_ids}

    # Build judge inputs: for each prompt, judge every response against that prompt's ground truth attribute
    conversations = []
    judge_attributes = []
    metadata = []  # (prompt_id, response_attribute, ground_truth_attribute)

    for pid in prompt_ids:
        gt_attr = gt_attrs[pid]
        prompt_df = df[df["prompt_id"] == pid]
        for _, row in prompt_df.iterrows():
            conversations.append([
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["response"]},
            ])
            judge_attributes.append(gt_attr)
            metadata.append((pid, row["attribute"], gt_attr))

    # Run judge
    judge: AttributeJudge = instantiate(cfg.judge)
    judge.load()
    scores = judge.judge_response_attribute(conversations, judge_attributes)
    judge.unload()

    # Analyze results per prompt
    results = pd.DataFrame(metadata, columns=["prompt_id", "response_attr", "gt_attr"])
    results["score"] = scores

    correct = 0
    correct_strict = 0
    margins = []

    for pid, group in results.groupby("prompt_id"):
        gt_attr = gt_attrs[pid]
        gt_score = group.loc[group["response_attr"] == gt_attr, "score"].values[0]
        mean_score = group["score"].mean()
        max_score = group["score"].max()

        correct += int(gt_score == max_score)
        num_at_max = (group["score"] == max_score).sum()
        correct_strict += int(gt_score == max_score and num_at_max == 1)
        margins.append(gt_score - mean_score)

    accuracy = correct / len(prompt_ids)
    accuracy_strict = correct_strict / len(prompt_ids)
    avg_margin = np.mean(margins)

    print(f"Accuracy (GT gets highest score): {accuracy:.3f} ({correct}/{len(prompt_ids)})")
    print(f"Accuracy strict (GT is sole max): {accuracy_strict:.3f} ({correct_strict}/{len(prompt_ids)})")
    print(f"Average margin over mean: {avg_margin:.3f}")


if __name__ == "__main__":
    main()
