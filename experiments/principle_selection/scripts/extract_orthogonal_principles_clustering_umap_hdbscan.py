import hydra
from omegaconf import DictConfig
from candidate_principles import candidate_principles
from hydra.utils import get_original_cwd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import umap
from sklearn.cluster import HDBSCAN


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    npy_path = Path(get_original_cwd()) / cfg.output_path
    scores_matrix = np.load(npy_path)
    num_conversations, num_principles = scores_matrix.shape
    principle_names = np.array(candidate_principles)

    # Standardize
    scores_std = (scores_matrix - scores_matrix.mean(axis=0)) / (scores_matrix.std(axis=0) + 1e-8)

    # Correlation matrix -> distance matrix -> UMAP embedding
    corr_matrix = np.corrcoef(scores_std, rowvar=False)
    distance_matrix = 1 - np.abs(corr_matrix)
    np.fill_diagonal(distance_matrix, 0)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    embedding = umap.UMAP(metric='precomputed', random_state=42).fit_transform(distance_matrix)

    abs_corr = np.abs(corr_matrix)

    def pick_representative(indices):
        if len(indices) == 1:
            return indices[0]
        sub = abs_corr[np.ix_(indices, indices)].copy()
        np.fill_diagonal(sub, 0)
        return indices[sub.mean(axis=1).argmax()]

    all_indices = np.arange(num_principles)
    best_labels = None
    for min_cluster_size in [3, 5, 8, 12]:
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size)
        labels = clusterer.fit_predict(embedding)
        n_clusters = len(set(labels) - {-1})
        noise_indices = np.where(labels == -1)[0]

        print(f"min_cluster_size={min_cluster_size}: {n_clusters} clusters, {len(noise_indices)} noise points")

        selected = []
        for cid in sorted(set(labels) - {-1}):
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
                'members': principle_names[indices],
                'intra': intra,
                'inter': inter,
            })
        selected.sort(key=lambda s: s['size'], reverse=True)

        print(f"  {'Representative':<40s} {'size':>4}  {'intra':>6}  {'inter':>6}")
        print("  " + "-" * 62)
        for s in selected:
            intra_str = f"{s['intra']:.3f}" if not np.isnan(s['intra']) else "   — "
            print(f"  {s['attribute']:<40s} {s['size']:>4}  {intra_str:>6}  {s['inter']:>6.3f}")
            for m in s['members']:
                marker = '*' if m == s['attribute'] else ' '
                print(f"    {marker} {m}")
        if len(noise_indices):
            print(f"  [noise: {', '.join(principle_names[noise_indices[:8]])}{'...' if len(noise_indices) > 8 else ''}]")
        print()

        if best_labels is None or n_clusters > len(set(best_labels) - {-1}):
            best_labels = labels

    # UMAP colored by best clustering
    unique_labels = sorted(set(best_labels))
    cmap = plt.get_cmap('tab20', len(unique_labels))
    fig, ax = plt.subplots(figsize=(20, 16))
    for i, label in enumerate(unique_labels):
        mask = best_labels == label
        color = 'lightgray' if label == -1 else cmap(i)
        ax.scatter(embedding[mask, 0], embedding[mask, 1], s=20, alpha=0.6, color=color)
    for i, name in enumerate(principle_names):
        ax.annotate(name, (embedding[i, 0], embedding[i, 1]), fontsize=7, alpha=0.7)
    ax.set_title('UMAP + HDBSCAN clustering of principles')
    ax.axis('off')
    plt.tight_layout()
    out_path = npy_path.with_name(npy_path.stem + '_umap_hdbscan.png')
    plt.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
