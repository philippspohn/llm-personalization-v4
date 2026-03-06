import hydra
from omegaconf import DictConfig
from candidate_principles import candidate_principles
from hydra.utils import get_original_cwd
import numpy as np
from pathlib import Path
from sklearn.cluster import HDBSCAN


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    npy_path = Path(get_original_cwd()) / cfg.output_path
    scores_matrix = np.load(npy_path)
    num_conversations, num_principles = scores_matrix.shape
    principle_names = np.array(candidate_principles)

    # Standardize
    scores_std = (scores_matrix - scores_matrix.mean(axis=0)) / (scores_matrix.std(axis=0) + 1e-8)

    # Correlation matrix -> distance matrix
    corr_matrix = np.corrcoef(scores_std, rowvar=False)
    distance_matrix = 1 - np.abs(corr_matrix)
    np.fill_diagonal(distance_matrix, 0)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2  # enforce symmetry

    abs_corr = np.abs(corr_matrix)

    def pick_representative(indices):
        if len(indices) == 1:
            return indices[0]
        sub = abs_corr[np.ix_(indices, indices)].copy()
        np.fill_diagonal(sub, 0)
        return indices[sub.mean(axis=1).argmax()]

    # Sweep min_cluster_size to see how many clusters / noise points result
    all_indices = np.arange(num_principles)
    for min_cluster_size in [3, 5, 8, 12]:
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size, metric='precomputed')
        labels = clusterer.fit_predict(distance_matrix)

        cluster_ids = sorted(set(labels) - {-1})
        noise_indices = np.where(labels == -1)[0]
        n_clusters = len(cluster_ids)

        print(f"min_cluster_size={min_cluster_size}: {n_clusters} clusters, {len(noise_indices)} noise points")

        selected = []
        for cid in cluster_ids:
            indices = np.where(labels == cid)[0]
            rep_idx = pick_representative(indices)
            sub = abs_corr[np.ix_(indices, indices)].copy()
            np.fill_diagonal(sub, np.nan)
            intra = np.nanmean(sub) if len(indices) > 1 else float('nan')
            other_indices = np.setdiff1d(all_indices, indices)
            inter = abs_corr[np.ix_(indices, other_indices)].mean()
            selected.append({
                'size': len(indices),
                'attribute': principle_names[rep_idx],
                'intra': intra,
                'inter': inter,
            })
        selected.sort(key=lambda s: s['size'], reverse=True)

        print(f"  {'Representative':<40s} {'size':>4}  {'intra':>6}  {'inter':>6}")
        print("  " + "-" * 62)
        for s in selected:
            intra_str = f"{s['intra']:.3f}" if not np.isnan(s['intra']) else "   — "
            print(f"  {s['attribute']:<40s} {s['size']:>4}  {intra_str:>6}  {s['inter']:>6.3f}")
        if len(noise_indices):
            print(f"  [noise: {', '.join(principle_names[noise_indices[:8]])}{'...' if len(noise_indices) > 8 else ''}]")
        print()


if __name__ == "__main__":
    main()
