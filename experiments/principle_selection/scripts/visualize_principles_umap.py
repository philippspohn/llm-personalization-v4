import hydra
from omegaconf import DictConfig
from candidate_principles import candidate_principles
from hydra.utils import get_original_cwd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import umap


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    npy_path = Path(get_original_cwd()) / cfg.output_path
    scores_matrix = np.load(npy_path)
    principle_names = candidate_principles

    # Correlation matrix -> distance matrix
    scores_std = (scores_matrix - scores_matrix.mean(axis=0)) / (scores_matrix.std(axis=0) + 1e-8)
    corr_matrix = np.corrcoef(scores_std, rowvar=False)
    distance_matrix = 1 - np.abs(corr_matrix)
    np.fill_diagonal(distance_matrix, 0)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2

    embedding = umap.UMAP(metric='precomputed', random_state=42).fit_transform(distance_matrix)

    fig, ax = plt.subplots(figsize=(20, 16))
    ax.scatter(embedding[:, 0], embedding[:, 1], s=20, alpha=0.6)
    for i, name in enumerate(principle_names):
        ax.annotate(name, (embedding[i, 0], embedding[i, 1]), fontsize=10, alpha=0.8)
    ax.set_title('UMAP of principles (distance = 1 - |correlation|)')
    ax.axis('off')
    plt.tight_layout()
    out_path = npy_path.with_name(npy_path.stem + '_umap.png')
    plt.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
