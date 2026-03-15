import argparse
import csv
import math
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

JUDGE_DISPLAY_NAMES = {
    "gemma_27b":              "Gemma 3 27B",
    "glm_4_5_air":            "GLM 4.5 Air",
    "glm_5":                  "GLM 5",
    "gpt_5_4":                "GPT 5.4",
    "gpt_oss_120b":           "gpt-oss-120b",
    "kimi_k2_5":              "Kimi-K2.5",
    "minimax_2_5":            "MiniMax-M2.5",
    "phi_4":                  "Phi-4",
    "phi_4_weighted":         "Phi-4 (prob. weighted)",
    "qwen3_5_27b":            "Qwen3.5 27B",
    "qwen3_5_27b_weighted":   "Qwen3.5 27B (prob. weighted)",
    "qwen3_5_35ba3b":         "Qwen3.5 35B A3B",
    "qwen3_5_122ba10b_bf":    "Qwen3.5 122B A10B",
    "step_3_5_flash":         "Step 3.5 Flash",
    "nemotron_3_super":       "Nemotron 3 Super",
}

def display_name(judge_key: str) -> str:
    return JUDGE_DISPLAY_NAMES.get(judge_key, judge_key)


def compute_rows(scores: np.ndarray, pairs: np.ndarray):
    """scores: (num_pairs, 2, conv_per_pair), pairs: (num_pairs, 2)
    Returns (rows, sum_scores_by_pair) where sum_scores_by_pair maps pair_label -> array.
    """
    rows = []
    all_sum_scores = []
    sum_scores_by_pair = {}

    for pair_idx, pair in enumerate(pairs):
        scores_a = scores[pair_idx, 0, :]
        scores_b = scores[pair_idx, 1, :]
        sum_scores = scores_a + scores_b
        pair_label = f"{pair[0]}/{pair[1]}"
        all_sum_scores.append(sum_scores)
        sum_scores_by_pair[pair_label] = sum_scores

        mean_sum = np.mean(sum_scores)
        std_sum = np.std(sum_scores)
        mean_l1 = np.mean(np.abs(sum_scores - 11))
        corr, pval = stats.pearsonr(scores_a, scores_b) if len(scores_a) >= 2 else (float("nan"), float("nan"))

        rows.append({
            "pair": pair_label,
            "mean_sum": mean_sum,
            "std_sum": std_sum,
            "mean_l1": mean_l1,
            "corr": corr,
            "pval": pval,
            "n": len(scores_a),
        })

    all_sum_scores_concat = np.concatenate(all_sum_scores)
    sum_scores_by_pair["overall"] = all_sum_scores_concat
    rows.append({
        "pair": "overall",
        "mean_sum": np.mean(all_sum_scores_concat),
        "std_sum": np.std(all_sum_scores_concat),
        "mean_l1": np.mean(np.abs(all_sum_scores_concat - 11)),
        "corr": float("nan"),
        "pval": float("nan"),
        "n": len(all_sum_scores_concat),
    })

    return rows, sum_scores_by_pair


def save_grid_plots(all_sum_scores: dict[str, dict[str, np.ndarray]], output_dir: str):
    """One PNG per pair (+ overall), all judges as subplots."""
    judge_names = list(all_sum_scores.keys())
    pair_labels = list(next(iter(all_sum_scores.values())).keys())

    n_judges = len(judge_names)
    ncols = min(4, n_judges)
    nrows = math.ceil(n_judges / ncols)
    bins = np.arange(-0.5, 21.5, 1)

    for pair_label in pair_labels:
        global_ymax = max(
            (np.histogram(all_sum_scores[j][pair_label], bins=bins)[0] / len(all_sum_scores[j][pair_label]) * 100).max()
            for j in judge_names if pair_label in all_sum_scores[j]
        )

        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
        axes = np.array(axes).reshape(nrows, ncols)

        for idx, judge_name in enumerate(judge_names):
            ax = axes[idx // ncols, idx % ncols]
            sum_scores = all_sum_scores[judge_name].get(pair_label)
            if sum_scores is None:
                ax.set_visible(False)
                continue
            weights = np.ones(len(sum_scores)) / len(sum_scores) * 100
            ax.hist(sum_scores, bins=bins, weights=weights, edgecolor="black")
            ax.set_title(display_name(judge_name), fontsize=9)
            ax.set_xlabel("Sum score", fontsize=8)
            ax.set_ylabel("%", fontsize=8)
            ax.set_xlim(0, 20)
            ax.set_ylim(0, global_ymax * 1.1)
            ax.axvline(11, color="red", linestyle="--", linewidth=1)

        for idx in range(n_judges, nrows * ncols):
            axes[idx // ncols, idx % ncols].set_visible(False)

        pair_safe = pair_label.replace("/", "_")
        fig.suptitle(f"Balance distribution — {pair_label}", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"grid_{pair_safe}_balance.png"), dpi=150)
        plt.close(fig)
        print(f"Saved grid_{pair_safe}_balance.png")


def save_heatmap(all_rows: dict, output_dir: str):
    judges = list(all_rows.keys())
    pairs = [r["pair"] for r in all_rows[judges[0]] if r["pair"] != "overall"]

    data = np.array([
        [next(r["corr"] for r in all_rows[j] if r["pair"] == p) for p in pairs]
        for j in judges
    ])

    cmap = LinearSegmentedColormap.from_list("gwr", ["green", "white", "red"])
    fig, ax = plt.subplots(figsize=(max(8, len(pairs) * 1.5), max(6, len(judges) * 0.5)))
    im = ax.imshow(data, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(pairs, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(judges)))
    ax.set_yticklabels([display_name(j) for j in judges], fontsize=9)
    ax.set_xlabel("Antonyms")
    ax.set_ylabel("Judge")
    ax.set_title("Discrimination: Correlation Between Antonym Pair Scores by Judge")
    plt.colorbar(im, ax=ax)
    for i in range(len(judges)):
        for j in range(len(pairs)):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(val) > 0.6 else "black")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "heatmap_correlation.png"), dpi=150)
    plt.close(fig)
    print("Saved heatmap_correlation.png")


def save_barchart(all_rows: dict, output_dir: str):
    overall = {
        judge: next(r["mean_l1"] for r in rows if r["pair"] == "overall")
        for judge, rows in all_rows.items()
    }
    sorted_judges = sorted(overall, key=overall.get, reverse=True)
    values = [overall[j] for j in sorted_judges]
    labels = [display_name(j) for j in sorted_judges]

    fig, ax = plt.subplots(figsize=(8, max(4, len(sorted_judges) * 0.4)))
    ax.barh(labels, values)
    ax.set_xlabel("Mean |sum - 11|")
    ax.set_ylabel("Judge")
    ax.set_title("Balance: Mean L1 Distance from 11 (lower = more balanced)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "barchart_mean_l1.png"), dpi=150)
    plt.close(fig)
    print("Saved barchart_mean_l1.png")


def save_csv(all_rows: dict, output_path: str):
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["judge", "pair", "n", "mean_sum", "std_sum", "mean_l1_from_11", "corr", "pval"])
        for judge_name, rows in all_rows.items():
            for row in rows:
                writer.writerow([
                    judge_name,
                    row["pair"],
                    row["n"],
                    round(row["mean_sum"], 4),
                    round(row["std_sum"], 4),
                    round(row["mean_l1"], 4),
                    "" if np.isnan(row["corr"]) else round(row["corr"], 4),
                    "" if np.isnan(row["pval"]) else round(row["pval"], 4),
                ])


def print_conv_counts(all_rows: dict):
    print("Conversations analyzed per principle pair:")
    for judge, rows in all_rows.items():
        n = next(r["n"] for r in rows if r["pair"] != "overall")
        print(f"  {judge}: {n}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores_dir",
        default=os.path.join(os.path.dirname(__file__), "../../../data/compare_judges"),
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(os.path.dirname(__file__), "../../../data/compare_judges/analysis"),
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    npz_files = sorted(glob.glob(os.path.join(args.scores_dir, "*.npz")))
    if not npz_files:
        print(f"No .npz files found in {args.scores_dir}")
        return

    all_rows = {}
    all_sum_scores = {}
    for path in npz_files:
        judge_name = os.path.splitext(os.path.basename(path))[0]
        data = np.load(path, allow_pickle=True)
        rows, sum_scores_by_pair = compute_rows(data["scores"], data["pairs"])
        all_rows[judge_name] = rows
        all_sum_scores[judge_name] = sum_scores_by_pair

    print_conv_counts(all_rows)
    save_csv(all_rows, os.path.join(args.output_dir, "table.csv"))
    save_grid_plots(all_sum_scores, args.output_dir)
    save_heatmap(all_rows, args.output_dir)
    save_barchart(all_rows, args.output_dir)


if __name__ == "__main__":
    main()
