import hydra
from omegaconf import DictConfig
from candidate_principles import candidate_principles
from hydra.utils import get_original_cwd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    scores_matrix = np.load(Path(get_original_cwd()) / cfg.output_path)
    num_conversations, num_principles = scores_matrix.shape
    principle_names = candidate_principles

    # Standardize
    scores_std = (scores_matrix - scores_matrix.mean(axis=0)) / (scores_matrix.std(axis=0) + 1e-8)

    # PCA via eigendecomposition of correlation matrix
    corr_matrix = np.corrcoef(scores_std, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    # Parallel analysis to choose k
    n_iterations = 100
    random_eigenvalues = np.zeros((n_iterations, num_principles))
    for i in range(n_iterations):
        random_data = np.random.normal(size=(num_conversations, num_principles))
        random_corr = np.corrcoef(random_data, rowvar=False)
        random_eigenvalues[i] = np.sort(np.linalg.eigh(random_corr)[0])[::-1]
    threshold = np.percentile(random_eigenvalues, 95, axis=0)
    k = int(np.sum(eigenvalues > threshold))

    # Loadings: eigenvectors scaled by sqrt(eigenvalue)
    loadings = eigenvectors[:, :k] * np.sqrt(eigenvalues[:k])

    # For each component, pick the attribute with highest absolute loading
    selected = []
    for comp_idx in range(k):
        best_attr_idx = np.argmax(np.abs(loadings[:, comp_idx]))
        selected.append({
            'component': comp_idx + 1,
            'eigenvalue': eigenvalues[comp_idx],
            'attribute': principle_names[best_attr_idx],
            'loading': loadings[best_attr_idx, comp_idx],
            'top5': sorted(
                [(principle_names[i], loadings[i, comp_idx]) for i in range(num_principles)],
                key=lambda x: abs(x[1]), reverse=True
            )[:5]
        })

    # Report
    cumulative_var = np.cumsum(eigenvalues) / eigenvalues.sum()
    print(f"Parallel analysis suggests k={k} components")
    print(f"These explain {cumulative_var[k-1]:.1%} of total variance\n")

    print(f"{'Comp':<6} {'Eigenvalue':<12} {'Var%':<8} {'CumVar%':<9} {'Representative':<35} {'Loading':<8}")
    print("-" * 90)
    for s in selected:
        idx = s['component'] - 1
        var_pct = eigenvalues[idx] / eigenvalues.sum() * 100
        cum_pct = cumulative_var[idx] * 100
        print(f"{s['component']:<6} {s['eigenvalue']:<12.2f} {var_pct:<8.1f} {cum_pct:<9.1f} {s['attribute']:<35} {s['loading']:<+8.3f}")

    print(f"\n{'='*90}")
    print("Top 5 loadings per component:\n")
    for s in selected:
        print(f"PC{s['component']}:")
        for name, loading in s['top5']:
            print(f"  {name:<40s} {loading:+.3f}")
        print()

    # Scree plot with parallel analysis threshold
    n_show = min(30, num_principles)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_show + 1), eigenvalues[:n_show], 'bo-', label='Data')
    plt.plot(range(1, n_show + 1), threshold[:n_show], 'r--', label='95th percentile (random)')
    plt.xlabel('Component')
    plt.ylabel('Eigenvalue')
    plt.title(f'Scree Plot with Parallel Analysis (k={k})')
    plt.axvline(x=k, color='gray', linestyle=':', alpha=0.5, label=f'k={k}')
    plt.legend()
    plt.tight_layout()
    plt.savefig('scree_parallel.png', dpi=150)
    print("Saved scree_parallel.png")

    # Save selected principles
    selected_names = [s['attribute'] for s in selected]
    print(f"\nFinal selected principles ({len(selected_names)}):")
    for name in selected_names:
        print(f"  - {name}")


if __name__ == "__main__":
    main()