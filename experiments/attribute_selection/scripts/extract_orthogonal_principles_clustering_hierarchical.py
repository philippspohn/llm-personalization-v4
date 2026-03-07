import hydra
from omegaconf import DictConfig
from candidate_principles import candidate_principles
from hydra.utils import get_original_cwd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    npy_path = Path(get_original_cwd()) / cfg.output_path
    scores_matrix = np.load(npy_path)
    num_conversations, num_principles = scores_matrix.shape
    principle_names = candidate_principles

    # Standardize
    scores_std = (scores_matrix - scores_matrix.mean(axis=0)) / (scores_matrix.std(axis=0) + 1e-8)

    # Correlation matrix -> distance matrix
    corr_matrix = np.corrcoef(scores_std, rowvar=False)
    distance_matrix = 1 - np.abs(corr_matrix)
    np.fill_diagonal(distance_matrix, 0)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2  # enforce symmetry
    linkage_matrix = linkage(squareform(distance_matrix), method='ward')

    def pick_representative(indices):
        """Principle with highest mean absolute correlation to other cluster members."""
        if len(indices) == 1:
            return indices[0]
        sub = np.abs(corr_matrix[np.ix_(indices, indices)])
        np.fill_diagonal(sub, 0)
        return indices[sub.mean(axis=1).argmax()]

    abs_corr = np.abs(corr_matrix)

    # Selections at different budgets
    for budget in [4, 8, 16, 32]:
        cluster_labels = fcluster(linkage_matrix, t=budget, criterion='maxclust')
        selected = []
        for cid in range(1, budget + 1):
            indices = np.where(cluster_labels == cid)[0]
            rep_idx = pick_representative(indices)
            # Intra-cluster: mean abs correlation within cluster (excluding diagonal)
            if len(indices) > 1:
                sub = abs_corr[np.ix_(indices, indices)].copy()
                np.fill_diagonal(sub, np.nan)
                intra = np.nanmean(sub)
            else:
                intra = float('nan')
            selected.append({
                'size': len(indices),
                'attribute': principle_names[rep_idx],
                'indices': indices,
                'intra': intra,
            })
        # Per-cluster inter: mean abs corr of each element to elements in other clusters
        all_indices = np.arange(num_principles)
        for s in selected:
            other_indices = np.setdiff1d(all_indices, s['indices'])
            s['inter'] = abs_corr[np.ix_(s['indices'], other_indices)].mean()

        selected.sort(key=lambda s: s['size'], reverse=True)

        print(f"Top {budget} principles:")
        print(f"  {'Representative':<40s} {'size':>4}  {'intra':>6}  {'inter':>6}")
        print("  " + "-" * 62)
        for s in selected:
            intra_str = f"{s['intra']:.3f}" if not np.isnan(s['intra']) else "   — "
            print(f"  {s['attribute']:<40s} {s['size']:>4}  {intra_str:>6}  {s['inter']:>6.3f}")
        print()

        groups = []
        for s in selected:
            indices = s['indices']
            rep_name = s['attribute']
            rep_idx = list(principle_names).index(rep_name)
            order = indices[np.argsort(-abs_corr[rep_idx][indices])]
            groups.append({'representative': rep_name, 'members': [principle_names[i] for i in order]})
        json_path = npy_path.with_name(npy_path.stem + f'_groups_hierarchical_k{budget}.json')
        json_path.write_text(json.dumps({'method': 'hierarchical', 'setting': f'k={budget}', 'k': budget, 'groups': groups}, indent=2))
        print(f"Saved {json_path}")

    # Dendrogram
    plt.figure(figsize=(20, 8))
    dendrogram(linkage_matrix, labels=principle_names, leaf_rotation=90, leaf_font_size=5)
    plt.title('Hierarchical Clustering of Principles')
    plt.tight_layout()
    out_path = npy_path.with_name(npy_path.stem + '_dendrogram.png')
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
