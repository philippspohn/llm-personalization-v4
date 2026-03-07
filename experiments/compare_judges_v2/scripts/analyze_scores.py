import argparse
import math
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def analyze_judge(judge_name: str, scores: np.ndarray, pairs: np.ndarray, output_dir: str):
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
        if len(scores_a) >= 2:
            corr, pval = stats.pearsonr(scores_a, scores_b)
        else:
            corr, pval = float("nan"), float("nan")

        rows.append({
            "pair": pair_label,
            "mean_sum": mean_sum,
            "std_sum": std_sum,
            "mean_l1": mean_l1,
            "corr": corr,
            "pval": pval,
        })

        fig, ax = plt.subplots()
        bins = np.arange(sum_scores.min() - 0.5, sum_scores.max() + 1.5, 1)
        ax.hist(sum_scores, bins=bins, edgecolor="black")
        ax.set_xlabel("Sum score")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{judge_name} — {pair_label}")
        fig.savefig(os.path.join(output_dir, f"{judge_name}_{pair[0]}_{pair[1]}_balance.png"))
        plt.close(fig)

    all_sum_scores_concat = np.concatenate(all_sum_scores)
    sum_scores_by_pair["overall"] = all_sum_scores_concat
    overall = {
        "pair": "overall",
        "mean_sum": np.mean(all_sum_scores_concat),
        "std_sum": np.std(all_sum_scores_concat),
        "mean_l1": np.mean(np.abs(all_sum_scores_concat - 11)),
        "corr": float("nan"),
        "pval": float("nan"),
    }
    rows.append(overall)

    fig, ax = plt.subplots()
    bins = np.arange(all_sum_scores_concat.min() - 0.5, all_sum_scores_concat.max() + 1.5, 1)
    ax.hist(all_sum_scores_concat, bins=bins, edgecolor="black")
    ax.set_xlabel("Sum score")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{judge_name} — Overall")
    fig.savefig(os.path.join(output_dir, f"{judge_name}_overall_balance.png"))
    plt.close(fig)

    return rows, sum_scores_by_pair


def save_grid_plots(all_sum_scores: dict[str, dict[str, np.ndarray]], output_dir: str):
    """all_sum_scores: {judge_name: {pair_label: sum_scores_array}}"""
    judge_names = list(all_sum_scores.keys())
    pair_labels = list(next(iter(all_sum_scores.values())).keys())  # includes "overall"

    n_judges = len(judge_names)
    ncols = min(4, n_judges)
    nrows = math.ceil(n_judges / ncols)

    bins = np.arange(-0.5, 21.5, 1)  # fixed x: 0-20

    for pair_label in pair_labels:
        # compute global y max using percentages so different sample sizes are comparable
        global_ymax = 0
        for judge_name in judge_names:
            sum_scores = all_sum_scores[judge_name].get(pair_label)
            if sum_scores is not None:
                counts, _ = np.histogram(sum_scores, bins=bins)
                global_ymax = max(global_ymax, (counts / len(sum_scores) * 100).max())

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
            ax.set_title(judge_name, fontsize=9)
            ax.set_xlabel("Sum score", fontsize=8)
            ax.set_ylabel("%", fontsize=8)
            ax.set_xlim(0, 20)
            ax.set_ylim(0, global_ymax * 1.1)
            ax.axvline(11, color="red", linestyle="--", linewidth=1)

        # hide unused axes
        for idx in range(n_judges, nrows * ncols):
            axes[idx // ncols, idx % ncols].set_visible(False)

        pair_safe = pair_label.replace("/", "_")
        fig.suptitle(f"Balance distribution — {pair_label}", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"grid_{pair_safe}_balance.png"), dpi=150)
        plt.close(fig)
        print(f"Saved grid_{pair_safe}_balance.png")


def print_table(all_rows: dict):
    header = f"{'Judge':<30} {'Pair':<20} {'mean':>6} {'std':>6} {'|sum-11|':>9} {'corr':>7} {'p':>7}"
    print(header)
    print("-" * len(header))
    for judge_name, rows in all_rows.items():
        for row in rows:
            corr_str = f"{row['corr']:7.3f}" if not np.isnan(row["corr"]) else "      -"
            pval_str = f"{row['pval']:7.3f}" if not np.isnan(row["pval"]) else "      -"
            print(
                f"{judge_name:<30} {row['pair']:<20} "
                f"{row['mean_sum']:6.2f} {row['std_sum']:6.2f} {row['mean_l1']:9.2f} "
                f"{corr_str} {pval_str}"
            )
        print()


def save_table(all_rows: dict, output_path: str):
    import csv
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["judge", "pair", "mean_sum", "std_sum", "mean_l1_from_11", "corr", "pval"])
        for judge_name, rows in all_rows.items():
            for row in rows:
                writer.writerow([
                    judge_name,
                    row["pair"],
                    round(row["mean_sum"], 4),
                    round(row["std_sum"], 4),
                    round(row["mean_l1"], 4),
                    "" if np.isnan(row["corr"]) else round(row["corr"], 4),
                    "" if np.isnan(row["pval"]) else round(row["pval"], 4),
                ])


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
        scores = data["scores"]   # (num_pairs, 2, conv_per_pair)
        pairs = data["pairs"]     # (num_pairs, 2)

        rows, sum_scores_by_pair = analyze_judge(judge_name, scores, pairs, args.output_dir)
        all_rows[judge_name] = rows
        all_sum_scores[judge_name] = sum_scores_by_pair

    print_table(all_rows)
    save_table(all_rows, os.path.join(args.output_dir, "table.csv"))
    save_grid_plots(all_sum_scores, args.output_dir)


if __name__ == "__main__":
    main()
