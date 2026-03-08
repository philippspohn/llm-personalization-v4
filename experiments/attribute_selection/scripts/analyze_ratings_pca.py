import hydra
from omegaconf import DictConfig
from hydra.utils import get_original_cwd
import numpy as np
from pathlib import Path

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    npy_path = Path(get_original_cwd()) / cfg.output_path
    scores_matrix = np.load(npy_path)
    num_conversations, num_principles = scores_matrix.shape
    principle_names = list(cfg.candidate_attributes)

    # Handle NaN
    if np.isnan(scores_matrix).any():
        col_mean = np.nanmean(scores_matrix, axis=0)
        for j in range(num_principles):
            mask = np.isnan(scores_matrix[:, j])
            if mask.any():
                scores_matrix[mask, j] = col_mean[j]

    # Standardize
    col_means = np.nanmean(scores_matrix, axis=0)
    col_stds = np.nanstd(scores_matrix, axis=0)
    col_std_safe = np.where(col_stds > 1e-10, col_stds, 1.0)
    scores_std = (scores_matrix - col_means) / col_std_safe

    # Handle zero variance
    zero_var = col_stds <= 1e-10
    if zero_var.any():
        np.random.seed(42)
        scores_std[:, zero_var] = np.random.randn(num_conversations, zero_var.sum())

    scores_std = np.asarray(scores_std, dtype=np.float64)
    corr_matrix = (scores_std.T @ scores_std) / (num_conversations - 1)
    np.fill_diagonal(corr_matrix, 1.0)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    # Parallel analysis — compute threshold at 95%
    n_iterations = 100
    random_eigenvalues = np.zeros((n_iterations, num_principles))
    for i in range(n_iterations):
        random_data = np.random.normal(size=(num_conversations, num_principles))
        random_corr = np.corrcoef(random_data, rowvar=False)
        random_eigenvalues[i] = np.sort(np.linalg.eigh(random_corr)[0])[::-1]
    
    threshold_95 = np.percentile(random_eigenvalues, 95, axis=0)
    k_95 = max(1, int(np.sum(eigenvalues > threshold_95)))

    # Calculate UNROTATED Loadings
    loadings = eigenvectors[:, :k_95] * np.sqrt(eigenvalues[:k_95])

    print(f"=== UNROTATED PCA (Parallel Analysis Ceiling: k={k_95}) ===\n")
    
    for comp_idx in range(k_95):
        # Calculate Eigenvalue (Variance Explained)
        ev = np.sum(loadings[:, comp_idx] ** 2)
        
        # Sort attributes by absolute loading magnitude
        top_attributes = sorted(
            [(principle_names[i], loadings[i, comp_idx]) for i in range(num_principles)],
            key=lambda x: abs(x[1]), reverse=True
        )[:5]

        print(f"Principal Component {comp_idx + 1} (Eigenvalue: {ev:.3f})")
        for name, loading in top_attributes:
            print(f"  {loading:>+7.3f} | {name}")
        print()

if __name__ == "__main__":
    main()