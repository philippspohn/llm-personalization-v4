import hydra
from omegaconf import DictConfig
from candidate_principles import candidate_principles
from hydra.utils import get_original_cwd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json


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
                                [np.sin(angle), np.cos(angle)]])
                rotation[:, [i, j]] = rotation[:, [i, j]] @ rot
        if np.max(np.abs(rotation - old)) < tol:
            break
    return loadings @ rotation


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    scores_matrix = np.load(Path(get_original_cwd()) / cfg.output_path)
    num_conversations, num_principles = scores_matrix.shape
    principle_names = candidate_principles

    # =========================================================================
    # Step 1: PCA on correlation matrix
    # =========================================================================
    scores_std = (scores_matrix - scores_matrix.mean(axis=0)) / (scores_matrix.std(axis=0) + 1e-8)
    corr_matrix = np.corrcoef(scores_std, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    # =========================================================================
    # Step 2: Parallel analysis to choose k
    # =========================================================================
    n_iterations = 100
    random_eigenvalues = np.zeros((n_iterations, num_principles))
    for i in range(n_iterations):
        random_data = np.random.normal(size=(num_conversations, num_principles))
        random_corr = np.corrcoef(random_data, rowvar=False)
        random_eigenvalues[i] = np.sort(np.linalg.eigh(random_corr)[0])[::-1]
    threshold = np.percentile(random_eigenvalues, 95, axis=0)
    k = int(np.sum(eigenvalues > threshold))
    cumulative_var = np.cumsum(eigenvalues) / eigenvalues.sum()
    print(f"Step 2: Parallel analysis suggests k={k} components")
    print(f"        These explain {cumulative_var[k-1]:.1%} of total variance")

    # =========================================================================
    # Step 3: Varimax rotation for interpretability
    # =========================================================================
    loadings = eigenvectors[:, :k] * np.sqrt(eigenvalues[:k])
    rotated_loadings = varimax(loadings)

    # =========================================================================
    # Step 4: Drop weak components (top loading below threshold)
    # =========================================================================
    LOADING_THRESHOLD = 0.5
    max_loadings = np.max(np.abs(rotated_loadings), axis=0)
    strong_mask = max_loadings >= LOADING_THRESHOLD
    n_strong = strong_mask.sum()
    print(f"\nStep 4: {n_strong}/{k} components have max loading >= {LOADING_THRESHOLD}")
    rotated_loadings = rotated_loadings[:, strong_mask]

    # =========================================================================
    # Step 5: Check for redundant components (high factor score correlation)
    #         If two components' score profiles correlate > threshold, merge
    # =========================================================================
    MERGE_THRESHOLD = 0.7
    factor_scores = scores_std @ np.linalg.pinv(rotated_loadings.T)  # (n_texts, n_components)
    score_corr = np.abs(np.corrcoef(factor_scores, rowvar=False))
    np.fill_diagonal(score_corr, 0)

    to_drop = set()
    n_components = rotated_loadings.shape[1]
    for i in range(n_components):
        for j in range(i + 1, n_components):
            if score_corr[i, j] > MERGE_THRESHOLD and j not in to_drop:
                # Keep the component with higher max loading
                max_i = np.max(np.abs(rotated_loadings[:, i]))
                max_j = np.max(np.abs(rotated_loadings[:, j]))
                drop = j if max_i >= max_j else i
                to_drop.add(drop)
                name_i = principle_names[np.argmax(np.abs(rotated_loadings[:, i]))]
                name_j = principle_names[np.argmax(np.abs(rotated_loadings[:, j]))]
                kept = name_i if drop == j else name_j
                dropped = name_j if drop == j else name_i
                print(f"Step 5: Merging '{dropped}' into '{kept}' (score corr={score_corr[i,j]:.2f})")

    keep_mask = [i for i in range(n_components) if i not in to_drop]
    rotated_loadings = rotated_loadings[:, keep_mask]
    n_final = rotated_loadings.shape[1]
    print(f"        {n_final} components remain after merging")

    # =========================================================================
    # Step 6: Select representative attribute per component
    # =========================================================================
    results = []
    for comp_idx in range(n_final):
        col = rotated_loadings[:, comp_idx]
        best_idx = np.argmax(np.abs(col))
        top5_indices = np.argsort(np.abs(col))[::-1][:5]
        results.append({
            'attribute': principle_names[best_idx],
            'loading': float(col[best_idx]),
            'top5': [(principle_names[i], float(col[i])) for i in top5_indices],
        })

    results.sort(key=lambda r: abs(r['loading']), reverse=True)
    missing = ['formal', 'casual', 'colloquial', 'sincere', 'optimistic', 
           'concise', 'structured', 'conversational', 'empathetic']
    for name in missing:
        if name in principle_names:
            idx = principle_names.index(name)
            loads = rotated_loadings[idx]
            top_comp = np.argmax(np.abs(loads))
            print(f"{name:30s} -> best component: RC{top_comp+1} ({results[top_comp]['attribute']}) loading={loads[top_comp]:+.3f}")

    sanity_pairs = [('formal', 'casual'), ('optimistic', 'pessimistic'), 
                ('concise', 'thorough'), ('humble', 'pompous'),
                ('structured', 'meandering')]
    for a, b in sanity_pairs:
        if a in principle_names and b in principle_names:
            i, j = principle_names.index(a), principle_names.index(b)
            print(f"corr({a}, {b}) = {corr_matrix[i,j]:+.3f}")
    # =========================================================================
    # Report
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"FINAL: {n_final} orthogonal style dimensions")
    print(f"{'='*70}\n")

    print(f"{'#':<4} {'Dimension':<40} {'Loading':<8}")
    print("-" * 55)
    for i, r in enumerate(results):
        print(f"{i+1:<4} {r['attribute']:<40} {r['loading']:+.3f}")

    print(f"\nDetailed loadings:\n")
    for i, r in enumerate(results):
        print(f"  Dimension {i+1}: {r['attribute']}")
        for name, loading in r['top5']:
            marker = " <--" if name == r['attribute'] else ""
            print(f"    {name:<45s} {loading:+.3f}{marker}")
        print()

    # =========================================================================
    # Scree plot
    # =========================================================================
    n_show = min(30, num_principles)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_show + 1), eigenvalues[:n_show], 'bo-', label='Data')
    plt.plot(range(1, n_show + 1), threshold[:n_show], 'r--', label='95th pctl (random)')
    plt.axvline(x=k, color='gray', linestyle=':', alpha=0.5, label=f'Parallel analysis: k={k}')
    plt.xlabel('Component')
    plt.ylabel('Eigenvalue')
    plt.title(f'Scree Plot with Parallel Analysis')
    plt.legend()
    plt.tight_layout()
    plt.savefig('scree_parallel.png', dpi=150)
    print("Saved scree_parallel.png")

    # =========================================================================
    # Save results
    # =========================================================================
    output = {
        'method': 'PCA + varimax rotation + parallel analysis',
        'n_texts': int(num_conversations),
        'n_candidate_attributes': int(num_principles),
        'k_parallel_analysis': int(k),
        'loading_threshold': LOADING_THRESHOLD,
        'merge_threshold': MERGE_THRESHOLD,
        'variance_explained': float(cumulative_var[k-1]),
        'dimensions': results,
    }
    with open('selected_principles.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("Saved selected_principles.json")


if __name__ == "__main__":
    main()