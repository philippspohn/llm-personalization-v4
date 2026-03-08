import hydra
from omegaconf import DictConfig
from hydra.utils import get_original_cwd
import numpy as np
from pathlib import Path

def varimax(loadings, max_iter=100, tol=1e-6):
    n, k = loadings.shape
    rotation = np.eye(k)
    for _ in range(max_iter):
        old = rotation.copy()
        for i in range(k):
            for j in range(i + 1, k):
                cols = loadings @ rotation[:, [i, j]]
                u = cols[:, 0] ** 2 - cols[:, 1] ** 2
                v = 2 * cols[:, 0] * cols[:, 1]
                num = 2 * n * (u @ v) - 2 * u.sum() * v.sum()
                den = n * ((u ** 2 - v ** 2).sum()) - (u.sum() ** 2 - v.sum() ** 2)
                angle = 0.25 * np.arctan2(num, den)
                rot = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle),  np.cos(angle)]])
                rotation[:, [i, j]] = rotation[:, [i, j]] @ rot
        if np.max(np.abs(rotation - old)) < tol:
            break
    return loadings @ rotation

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
    k_ceiling = max(1, int(np.sum(eigenvalues > threshold_95)))

    # Determine which k's to test (powers of 2 up to ceiling, plus the ceiling)
    k_targets = [2**i for i in range(1, int(np.log2(k_ceiling)) + 1)]
    if k_ceiling not in k_targets:
        k_targets.append(k_ceiling)
    
    # Ensure they are unique and sorted
    k_targets = sorted(list(set([k for k in k_targets if k <= k_ceiling])))

    print(f"Parallel Analysis determined a ceiling of k={k_ceiling}\n")

    for k in k_targets:
        print(f"============================================================")
        print(f"=== VARIMAX ROTATION WITH k={k} ===")
        print(f"============================================================\n")

        # 1. Slice the unrotated components up to k
        loadings = eigenvectors[:, :k] * np.sqrt(eigenvalues[:k])
        
        # 2. Apply Varimax
        rotated = varimax(loadings)

        # 3. Calculate SSL and extract top attributes
        components_info = []
        for comp_idx in range(k):
            ssl = np.sum(rotated[:, comp_idx] ** 2)
            
            top_attributes = sorted(
                [(principle_names[i], rotated[i, comp_idx]) for i in range(num_principles)],
                key=lambda x: abs(x[1]), reverse=True
            )[:5]
            
            components_info.append({
                'ssl': ssl,
                'top_attributes': top_attributes
            })

        # Sort the rotated components by SSL (Eigenvalue equivalent) descending
        components_info.sort(key=lambda x: x['ssl'], reverse=True)

        for idx, info in enumerate(components_info):
            print(f"Rotated Component {idx + 1} (SSL: {info['ssl']:.3f})")
            for name, loading in info['top_attributes']:
                print(f"  {loading:>+7.3f} | {name}")
            print()

if __name__ == "__main__":
    main()