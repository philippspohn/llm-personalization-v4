import json
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, get_original_cwd
from llm_personalization.judge import ParsedRatingOpenRouterJudge
from llm_personalization.data.load_ultrachat import load_ultrachat_conversations


OPPOSITE_PAIRS = [
    ("formal", "casual"),
    ("humorous", "serious"),
    ("concise", "verbose"),
]
NUM_REPEATS = 5


def collect_scores(
    judge: ParsedRatingOpenRouterJudge,
    conversations: list[list[dict[str, str]]],
    principles: list[str],
    num_repeats: int,
) -> np.ndarray:
    """Run the judge num_repeats times on every (conversation, principle) pair.

    Returns an array of shape (num_conversations, num_principles, num_repeats).
    """
    conversations_to_judge = []
    principles_to_judge = []

    for conversation in conversations:
        for principle in principles:
            for _ in range(num_repeats):
                conversations_to_judge.append(conversation)
                principles_to_judge.append(principle)

    raw_scores = judge.judge_principle(
        conversations=conversations_to_judge,
        principles=principles_to_judge,
    )

    return np.array(raw_scores).reshape(len(conversations), len(principles), num_repeats)


def compute_metrics(
    scores: np.ndarray,
    principles: list[str],
    judge_name: str,
) -> dict:
    """Compute all comparison metrics from a (C, P, R) score array."""
    num_conversations, num_principles, num_repeats = scores.shape

    metrics: dict = {"judge": judge_name}

    # --- 1. Intra-judge consistency (variation across repeats) ---
    repeat_std = scores.std(axis=2)  # (C, P)
    metrics["consistency"] = {}
    for p_idx, principle in enumerate(principles):
        per_principle = repeat_std[:, p_idx]
        metrics["consistency"][principle] = {
            "mean_std": float(per_principle.mean()),
            "max_std": float(per_principle.max()),
        }
    metrics["consistency"]["overall_mean_std"] = float(repeat_std.mean())

    # --- 2. Opposite-pair sum stability ---
    mean_scores = scores.mean(axis=2)  # (C, P) — average over repeats
    metrics["opposite_pairs"] = {}
    for pos, neg in OPPOSITE_PAIRS:
        if pos not in principles or neg not in principles:
            continue
        pos_idx = principles.index(pos)
        neg_idx = principles.index(neg)
        pair_sums = mean_scores[:, pos_idx] + mean_scores[:, neg_idx]  # (C,)
        metrics["opposite_pairs"][f"{pos}+{neg}"] = {
            "mean_sum": float(pair_sums.mean()),
            "std_sum": float(pair_sums.std()),
            "min_sum": float(pair_sums.min()),
            "max_sum": float(pair_sums.max()),
            "per_conversation": pair_sums.tolist(),
        }

    # --- 3. Per-principle score distribution ---
    metrics["score_distribution"] = {}
    for p_idx, principle in enumerate(principles):
        vals = scores[:, p_idx, :].flatten()
        metrics["score_distribution"][principle] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }

    return metrics


def print_summary(metrics: dict, principles: list[str]) -> None:
    judge_name = metrics["judge"]
    print(f"\n{'=' * 60}")
    print(f"  Judge: {judge_name}")
    print(f"{'=' * 60}")

    print("\n--- Consistency (std across repeats, lower = more consistent) ---")
    for principle in principles:
        m = metrics["consistency"][principle]
        print(f"  {principle:>12s}:  mean_std = {m['mean_std']:.3f}  max_std = {m['max_std']:.3f}")
    print(f"  {'overall':>12s}:  mean_std = {metrics['consistency']['overall_mean_std']:.3f}")

    print("\n--- Opposite-pair sums (should be ~11 for a 1-10 scale) ---")
    for pair_key, m in metrics["opposite_pairs"].items():
        print(f"  {pair_key:>20s}:  mean = {m['mean_sum']:.2f}  std = {m['std_sum']:.2f}  "
              f"range = [{m['min_sum']:.1f}, {m['max_sum']:.1f}]")

    print("\n--- Score distribution per principle ---")
    for principle in principles:
        m = metrics["score_distribution"][principle]
        print(f"  {principle:>12s}:  mean = {m['mean']:.2f}  std = {m['std']:.2f}  "
              f"range = [{m['min']:.0f}, {m['max']:.0f}]")

    print()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    principles = []
    for pos, neg in OPPOSITE_PAIRS:
        principles.extend([pos, neg])

    conversations = load_ultrachat_conversations(
        split="train_sft",
        limit=20,
        seed=42,
    )

    orig_cwd = Path(get_original_cwd())
    judge_config_dir = orig_cwd / "experiments" / "compare_judges" / "configs" / "judge"
    output_dir = orig_cwd / "experiments" / "compare_judges" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    failed_judges: list[tuple[str, str]] = []
    results_path = output_dir / "all_results.json"

    for judge_name in cfg.judges:
        print(f"\n>>> Loading judge: {judge_name}")
        try:
            judge_cfg = OmegaConf.load(judge_config_dir / f"{judge_name}.yaml")
            judge: ParsedRatingOpenRouterJudge = instantiate(judge_cfg)
            judge.load()
        except Exception as e:
            print(f"!!! Failed to load judge {judge_name}: {e}")
            failed_judges.append((judge_name, str(e)))
            continue

        try:
            scores = collect_scores(judge, conversations, principles, NUM_REPEATS)

            np.save(output_dir / f"scores_{judge_name}.npy", scores)

            metrics = compute_metrics(scores, principles, judge_name)
            all_results.append(metrics)

            print_summary(metrics, principles)
        except Exception as e:
            print(f"!!! Judge {judge_name} failed during scoring: {e}")
            failed_judges.append((judge_name, str(e)))
        finally:
            judge.unload()

        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to {results_path}")

    # --- Cross-judge agreement summary ---
    if len(all_results) > 1:
        print(f"\n{'=' * 60}")
        print("  Cross-judge agreement (deviation from grand mean)")
        print(f"{'=' * 60}")

        all_mean_scores = []
        judge_names = []
        for result in all_results:
            judge_names.append(result["judge"])
            means = [result["score_distribution"][p]["mean"] for p in principles]
            all_mean_scores.append(means)

        all_mean_scores = np.array(all_mean_scores)  # (J, P)
        grand_mean = all_mean_scores.mean(axis=0)  # (P,)

        for j_idx, jname in enumerate(judge_names):
            deviations = all_mean_scores[j_idx] - grand_mean
            mad = np.abs(deviations).mean()
            print(f"\n  {jname}:  mean |deviation| = {mad:.2f}")
            for p_idx, principle in enumerate(principles):
                print(f"    {principle:>12s}:  judge_mean = {all_mean_scores[j_idx, p_idx]:.2f}  "
                      f"grand_mean = {grand_mean[p_idx]:.2f}  "
                      f"delta = {deviations[p_idx]:+.2f}")

    if failed_judges:
        print(f"\n{'=' * 60}")
        print(f"  {len(failed_judges)} judge(s) FAILED:")
        print(f"{'=' * 60}")
        for name, err in failed_judges:
            print(f"  - {name}: {err}")


if __name__ == "__main__":
    main()
