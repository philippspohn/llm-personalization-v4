import hydra
from omegaconf import DictConfig
from hydra.utils import get_original_cwd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    npy_path = Path(get_original_cwd()) / cfg.output_path
    scores_matrix = np.load(npy_path)
    principle_names = list(cfg.candidate_attributes)
    n = len(principle_names)

    score_min = int(scores_matrix.min())
    score_max = int(scores_matrix.max())
    bins = np.arange(score_min, score_max + 2) - 0.5  # bin edges centred on integers

    ncols = 20
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.6, nrows * 1.4))
    axes = axes.flatten()

    for i, name in enumerate(principle_names):
        ax = axes[i]
        col = scores_matrix[:, i]
        counts, _ = np.histogram(col, bins=bins)
        xs = np.arange(score_min, score_max + 1)
        ax.bar(xs, counts, width=0.8, color='steelblue', linewidth=0)
        ax.set_title(name, fontsize=4.5, pad=2)
        ax.set_xlim(bins[0], bins[-1])
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.tick_params(axis='both', labelsize=3.5, length=2, pad=1)
        ax.spines[['top', 'right']].set_visible(False)

    # Hide unused subplots
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Score distributions per principle', fontsize=12, y=1.002)
    plt.tight_layout(h_pad=1.2, w_pad=0.5)

    out_path = npy_path.with_name(npy_path.stem + '_score_distributions.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
