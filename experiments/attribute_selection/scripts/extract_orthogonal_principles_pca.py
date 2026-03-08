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
    npy_path = Path(get_original_cwd()) / cfg.output_path
    scores_matrix = np.load(npy_path)
    num_conversations, num_principles = scores_matrix.shape
    principle_names = list(cfg.candidate_attributes)
    print(f"First 5 principles: {principle_names[:5]}")
    if len(principle_names) != num_principles:
        raise ValueError(f"Number of principles in config ({len(principle_names)}) does not match number of principles in scores matrix ({num_principles})")

    # Handle NaN (e.g. from parse failures): replace with column mean
    if np.isnan(scores_matrix).any():
        col_mean = np.nanmean(scores_matrix, axis=0)
        for j in range(num_principles):
            mask = np.isnan(scores_matrix[:, j])
            if mask.any():
                scores_matrix[mask, j] = col_mean[j]
        print(f"Warning: Replaced NaN values with column means")

    # Standardize (avoid div-by-zero: use 1.0 when std is negligible)
    col_std = np.nanstd(scores_matrix, axis=0)
    col_std = np.where(col_std > 1e-10, col_std, 1.0)
    scores_std = (scores_matrix - np.nanmean(scores_matrix, axis=0)) / col_std
    # Replace zero-variance columns (originally constant) with small noise so they don't break correlation
    zero_var = scores_matrix.std(axis=0) <= 1e-10
    if zero_var.any():
        zero_names = [principle_names[i] for i in np.where(zero_var)[0]]
        print(f"Warning: {zero_var.sum()} principle(s) have zero variance: {zero_names[:5]}{'...' if len(zero_names) > 5 else ''}")
        np.random.seed(42)
        scores_std[:, zero_var] = np.random.randn(num_conversations, zero_var.sum())  # unit variance for corr matrix

    # Correlation matrix from standardized data: corr = (X.T @ X) / (n-1), avoids np.corrcoef div-by-zero
    scores_std = np.asarray(scores_std, dtype=np.float64)
    n_eff = num_conversations - 1
    corr_matrix = (scores_std.T @ scores_std) / n_eff
    np.fill_diagonal(corr_matrix, 1.0)  # ensure diagonal is exactly 1
    if np.isnan(corr_matrix).any() or np.isinf(corr_matrix).any():
        raise ValueError("Correlation matrix contains NaN/Inf. Check for invalid or constant columns in scores.")
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
    k = max(1, int(np.sum(eigenvalues > threshold)))  # at least 1 component

    # Loadings: eigenvectors scaled by sqrt(eigenvalue)
    loadings = eigenvectors[:, :k] * np.sqrt(eigenvalues[:k])

    # Varimax rotation
    rotated_loadings = varimax(loadings)

    # For each component, pick the attribute with highest absolute loading
    selected = []
    for comp_idx in range(k):
        best_attr_idx = np.argmax(np.abs(rotated_loadings[:, comp_idx]))
        selected.append({
            'component': comp_idx + 1,
            'eigenvalue': float(np.sum(rotated_loadings[:, comp_idx] ** 2)),
            'attribute': principle_names[best_attr_idx],
            'loading': rotated_loadings[best_attr_idx, comp_idx],
            'top5': sorted(
                [(principle_names[i], rotated_loadings[i, comp_idx]) for i in range(num_principles)],
                key=lambda x: abs(x[1]), reverse=True
            )[:5]
        })

    # Sort by max absolute loading (descending) to rank by interpretability
    selected.sort(key=lambda s: abs(s['loading']), reverse=True)
    for i, s in enumerate(selected):
        s['rank'] = i + 1

    # Report
    cumulative_var = np.cumsum(eigenvalues) / eigenvalues.sum()
    print(f"Parallel analysis suggests k={k} components")
    print(f"These explain {cumulative_var[k-1]:.1%} of total variance\n")

    print(f"{'Rank':<6} {'Eigenvalue':<12} {'Representative':<40} {'Loading':<8}")
    print("-" * 70)
    for s in selected:
        print(f"{s['rank']:<6} {s['eigenvalue']:<12.2f} {s['attribute']:<40} {s['loading']:<+8.3f}")

    print(f"\n{'='*70}")
    print("Top 5 loadings per rotated component:\n")
    for s in selected:
        print(f"RC{s['rank']} (representative: {s['attribute']}):")
        for name, loading in s['top5']:
            print(f"  {name:<45s} {loading:+.3f}")
        print()

    # Antonym correlations (robustness check)
    antonym_pairs = [
        ('formal', 'casual'),
        ('concise', 'verbose'),
        ('optimistic', 'pessimistic'),
        ('emotionally neutral', 'appeals to emotion'),
        ('acknowledges uncertainty', 'absolute certainty'),
    ]
    name_to_idx = {name: i for i, name in enumerate(principle_names)}
    print("Antonym correlations (negative = judge discriminates well):")
    for a, b in antonym_pairs:
        if a in name_to_idx and b in name_to_idx:
            c = corr_matrix[name_to_idx[a], name_to_idx[b]]
            print(f"  {a} / {b}: {c:+.3f}")
            col_a = scores_matrix[:, name_to_idx[a]]
            col_b = scores_matrix[:, name_to_idx[b]]
            print(f"    {a}: mean={col_a.mean():.1f} std={col_a.std():.1f} range=[{col_a.min():.0f},{col_a.max():.0f}]")
            print(f"    {b}: mean={col_b.mean():.1f} std={col_b.std():.1f} range=[{col_b.min():.0f},{col_b.max():.0f}]")
            pair_sum = col_a + col_b
            print(f"    sum: mean={pair_sum.mean():.1f} std={pair_sum.std():.1f} range=[{pair_sum.min():.0f},{pair_sum.max():.0f}]")
    print()

    # Selections at different budgets
    for budget in [4, 8, 16, 32]:
        subset = selected[:budget]
        print(f"Top {len(subset)} principles:")
        for s in subset:
            print(f"  {s['rank']:>2}. {s['attribute']}")
        print()

    # Save JSON for each budget
    def build_pca_groups(budget):
        rl = rotated_loadings[:, :budget]
        assignments = np.argmax(np.abs(rl), axis=1)
        groups = []
        for comp_idx in range(budget):
            indices = np.where(assignments == comp_idx)[0]
            if len(indices) == 0:
                continue
            order = indices[np.argsort(-np.abs(rl[indices, comp_idx]))]
            rep_loading = float(np.abs(rl[order[0], comp_idx]))
            groups.append({
                'representative': principle_names[order[0]],
                'members': [principle_names[i] for i in order],
                '_rep_loading': rep_loading,
            })
        groups.sort(key=lambda g: -g['_rep_loading'])
        for g in groups:
            del g['_rep_loading']
        return groups

    for budget in [4, 8, 16, k, 32]:
        if budget > k:
            continue
        groups = build_pca_groups(budget)
        out = {'method': 'pca', 'setting': f'k={budget}', 'k': budget, 'groups': groups}
        json_path = npy_path.with_name(npy_path.stem + f'_groups_pca_k{budget}.json')
        json_path.write_text(json.dumps(out, indent=2))
        print(f"Saved {json_path}")

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
    plt.savefig(npy_path.with_name(npy_path.stem + '_scree.png'), dpi=150)
    print(f"Saved {npy_path.with_name(npy_path.stem + '_scree.png')}")

    # Save
    np.savez(
        npy_path.with_name(npy_path.stem + '_pca_results.npz'),
        eigenvalues=eigenvalues,
        rotated_loadings=rotated_loadings,
        selected_names=[s['attribute'] for s in selected],
        selected_loadings=[s['loading'] for s in selected],
    )
    print(f"Saved {npy_path.with_name(npy_path.stem + '_pca_results.npz')}")


if __name__ == "__main__":
    main()