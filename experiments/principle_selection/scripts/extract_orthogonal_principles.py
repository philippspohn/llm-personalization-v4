import hydra
from omegaconf import DictConfig
from candidate_principles import candidate_principles
from hydra.utils import get_original_cwd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    scores_matrix = np.load(Path(get_original_cwd()) / cfg.output_path)
    num_conversations, num_principles = scores_matrix.shape
    principle_names = candidate_principles


    # Standardize
    scores_std = (scores_matrix - scores_matrix.mean(axis=0)) / (scores_matrix.std(axis=0) + 1e-8)

    # Correlation matrix
    corr_matrix = np.corrcoef(scores_std, rowvar=False)

    # PCA via eigendecomposition of correlation matrix
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
    # eigh returns ascending order, flip to descending
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    # Scree plot
    plt.figure(figsize=(12, 5))
    plt.plot(range(1, min(50, len(eigenvalues)) + 1), eigenvalues[:50], 'bo-')
    plt.axhline(y=1, color='r', linestyle='--', label='Kaiser criterion')
    plt.xlabel('Component number')
    plt.ylabel('Eigenvalue')
    plt.title('Scree Plot (first 50 components)')
    plt.legend()
    plt.savefig('scree_plot.png', dpi=150)
    print("Saved scree_plot.png")

    # Parallel analysis
    n_iterations = 50
    random_eigenvalues = np.zeros((n_iterations, num_principles))
    for i in range(n_iterations):
        random_data = np.random.normal(size=(num_conversations, num_principles))
        random_corr = np.corrcoef(random_data, rowvar=False)
        rand_ev = np.linalg.eigh(random_corr)[0][::-1]
        random_eigenvalues[i] = rand_ev
    threshold = np.percentile(random_eigenvalues, 95, axis=0)
    n_factors = np.sum(eigenvalues > threshold)
    print(f"Parallel analysis suggests {n_factors} factors")
    print(f"Top 20 eigenvalues: {eigenvalues[:20].round(2)}")
    print(f"Top 20 thresholds:  {threshold[:20].round(2)}")

    # Variance explained
    cumulative_var = np.cumsum(eigenvalues) / eigenvalues.sum()
    print(f"Variance explained by {n_factors} factors: {cumulative_var[n_factors-1]:.1%}")

    # Loadings: eigenvectors scaled by sqrt(eigenvalue)
    loadings = eigenvectors[:, :n_factors] * np.sqrt(eigenvalues[:n_factors])

    # Varimax rotation
    def varimax(loadings, max_iter=100, tol=1e-6):
        """Simple varimax rotation."""
        n, k = loadings.shape
        rotation = np.eye(k)
        for _ in range(max_iter):
            old_rotation = rotation.copy()
            for i in range(k):
                for j in range(i + 1, k):
                    cols = loadings @ rotation[:, [i, j]]
                    u = cols[:, 0] ** 2 - cols[:, 1] ** 2
                    v = 2 * cols[:, 0] * cols[:, 1]
                    num = 2 * n * (u @ v) - 2 * u.sum() * v.sum()
                    den = n * ((u ** 2 - v ** 2).sum()) - (u.sum() ** 2 - v.sum() ** 2)
                    angle = 0.25 * np.arctan2(num, den)
                    rot = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])
                    rotation[:, [i, j]] = rotation[:, [i, j]] @ rot
            if np.max(np.abs(rotation - old_rotation)) < tol:
                break
        return loadings @ rotation, rotation

    rotated_loadings, _ = varimax(loadings)

    # Top principles per factor
    print(f"\n=== {n_factors} factors found ===\n")
    selected = []
    for factor_idx in range(n_factors):
        fl = rotated_loadings[:, factor_idx]
        top_indices = np.argsort(np.abs(fl))[::-1][:5]
        print(f"Factor {factor_idx + 1}:")
        for idx in top_indices:
            print(f"  {principle_names[idx]:40s} loading={fl[idx]:+.3f}")
        best_idx = np.argmax(np.abs(fl))
        selected.append(best_idx)
        print()

    print("=== Selected representatives ===")
    for i, idx in enumerate(selected):
        print(f"  Factor {i+1}: {principle_names[idx]}")

    # Hierarchical clustering for comparison
    distance_matrix = 1 - np.abs(corr_matrix)
    np.fill_diagonal(distance_matrix, 0)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2  # force symmetry
    condensed_dist = squareform(distance_matrix)
    linkage_matrix = linkage(condensed_dist, method='ward')

    plt.figure(figsize=(20, 10))
    dendrogram(linkage_matrix, labels=principle_names, leaf_rotation=90, leaf_font_size=4)
    plt.title('Hierarchical Clustering of Principles')
    plt.tight_layout()
    plt.savefig('dendrogram.png', dpi=150)
    print("\nSaved dendrogram.png")

    # Compare: cut tree at n_factors clusters
    cluster_labels = fcluster(linkage_matrix, t=n_factors, criterion='maxclust')
    for cid in range(1, n_factors + 1):
        members = [principle_names[i] for i in range(num_principles) if cluster_labels[i] == cid]
        print(f"\nCluster {cid} ({len(members)} members): {members[:8]}{'...' if len(members) > 8 else ''}")


if __name__ == "__main__":
    main()