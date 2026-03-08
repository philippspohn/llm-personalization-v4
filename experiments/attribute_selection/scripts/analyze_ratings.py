import hydra
from omegaconf import DictConfig
from hydra.utils import get_original_cwd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import umap


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


def compute_entropy(col):
    col = col[~np.isnan(col)]
    _, counts = np.unique(col.round().astype(int), return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p + 1e-12)))


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    npy_path = Path(get_original_cwd()) / cfg.output_path
    scores_matrix = np.load(npy_path)
    num_conversations, num_principles = scores_matrix.shape
    principle_names = list(cfg.candidate_attributes)

    if len(principle_names) != num_principles:
        raise ValueError(
            f"Number of principles in config ({len(principle_names)}) does not match "
            f"number of principles in scores matrix ({num_principles})"
        )

    # Handle NaN
    if np.isnan(scores_matrix).any():
        col_mean = np.nanmean(scores_matrix, axis=0)
        for j in range(num_principles):
            mask = np.isnan(scores_matrix[:, j])
            if mask.any():
                scores_matrix[mask, j] = col_mean[j]
        print("Warning: Replaced NaN values with column means")

    # Per-attribute stats
    col_means = np.nanmean(scores_matrix, axis=0)
    col_stds = np.nanstd(scores_matrix, axis=0)
    col_entropies = np.array([compute_entropy(scores_matrix[:, j]) for j in range(num_principles)])

    # Standardize
    col_std_safe = np.where(col_stds > 1e-10, col_stds, 1.0)
    scores_std = (scores_matrix - col_means) / col_std_safe

    zero_var = col_stds <= 1e-10
    if zero_var.any():
        zero_names = [principle_names[i] for i in np.where(zero_var)[0]]
        print(f"Warning: {zero_var.sum()} zero-variance principle(s): {zero_names[:5]}")
        np.random.seed(42)
        scores_std[:, zero_var] = np.random.randn(num_conversations, zero_var.sum())

    scores_std = np.asarray(scores_std, dtype=np.float64)
    corr_matrix = (scores_std.T @ scores_std) / (num_conversations - 1)
    np.fill_diagonal(corr_matrix, 1.0)
    if np.isnan(corr_matrix).any() or np.isinf(corr_matrix).any():
        raise ValueError("Correlation matrix contains NaN/Inf.")

    name_to_idx = {name: i for i, name in enumerate(principle_names)}

    # =========================================================================
    # 1. Antonym sanity check
    # =========================================================================
    antonym_pairs = [
        ('formal', 'casual'),
        ('concise', 'verbose'),
        ('optimistic', 'pessimistic'),
        ('emotionally neutral', 'appeals to emotion'),
        ('acknowledges uncertainty', 'speaks in absolutes'),
    ]
    print("Antonym correlations (negative = judge discriminates well):")
    for a, b in antonym_pairs:
        if a in name_to_idx and b in name_to_idx:
            c = corr_matrix[name_to_idx[a], name_to_idx[b]]
            col_a = scores_matrix[:, name_to_idx[a]]
            col_b = scores_matrix[:, name_to_idx[b]]
            pair_sum = col_a + col_b
            print(f"  {a} / {b}: corr={c:+.3f}")
            print(f"    {a}: mean={col_a.mean():.1f}  std={col_a.std():.1f}  range=[{col_a.min():.0f},{col_a.max():.0f}]")
            print(f"    {b}: mean={col_b.mean():.1f}  std={col_b.std():.1f}  range=[{col_b.min():.0f},{col_b.max():.0f}]")
            print(f"    sum: mean={pair_sum.mean():.1f}  std={pair_sum.std():.1f}  range=[{pair_sum.min():.0f},{pair_sum.max():.0f}]")
    print()

    # =========================================================================
    # 2. Score distributions grid
    # =========================================================================
    score_min = int(np.nanmin(scores_matrix))
    score_max = int(np.nanmax(scores_matrix))
    bins = np.arange(score_min, score_max + 2) - 0.5

    ncols = 20
    nrows = (num_principles + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.6, nrows * 1.4))
    axes = axes.flatten()

    for i, name in enumerate(principle_names):
        ax = axes[i]
        counts, _ = np.histogram(scores_matrix[:, i], bins=bins)
        ax.bar(np.arange(score_min, score_max + 1), counts, width=0.8, color='steelblue', linewidth=0)
        ax.set_title(name, fontsize=4.5, pad=2)
        ax.set_xlim(bins[0], bins[-1])
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.tick_params(axis='both', labelsize=3.5, length=2, pad=1)
        ax.spines[['top', 'right']].set_visible(False)

    for j in range(num_principles, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Score distributions per principle', fontsize=12, y=1.002)
    plt.tight_layout(h_pad=1.2, w_pad=0.5)
    dist_path = npy_path.with_name(npy_path.stem + '_score_distributions.png')
    plt.savefig(dist_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {dist_path}")

    # =========================================================================
    # 3. PCA + Varimax
    # =========================================================================
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    # Parallel analysis — compute threshold at 95%
    n_show_rcs = min(50, num_principles)
    n_iterations = 100
    random_eigenvalues = np.zeros((n_iterations, num_principles))
    for i in range(n_iterations):
        random_data = np.random.normal(size=(num_conversations, num_principles))
        random_corr = np.corrcoef(random_data, rowvar=False)
        random_eigenvalues[i] = np.sort(np.linalg.eigh(random_corr)[0])[::-1]
    threshold_95 = np.percentile(random_eigenvalues, 95, axis=0)
    k_95 = max(1, int(np.sum(eigenvalues > threshold_95)))

    loadings = eigenvectors[:, :n_show_rcs] * np.sqrt(eigenvalues[:n_show_rcs])
    rotated_loadings = varimax(loadings)

    selected = []
    for comp_idx in range(n_show_rcs):
        best_idx = np.argmax(np.abs(rotated_loadings[:, comp_idx]))
        top5 = sorted(
            [(principle_names[i], rotated_loadings[i, comp_idx]) for i in range(num_principles)],
            key=lambda x: abs(x[1]), reverse=True
        )[:5]
        selected.append({
            'component': comp_idx + 1,
            'eigenvalue': float(np.sum(rotated_loadings[:, comp_idx] ** 2)),
            'attribute': principle_names[best_idx],
            'loading': float(rotated_loadings[best_idx, comp_idx]),
            'top5': top5,
        })

    selected.sort(key=lambda s: abs(s['loading']), reverse=True)
    for i, s in enumerate(selected):
        s['rank'] = i + 1

    cumulative_var = np.cumsum(eigenvalues) / eigenvalues.sum()

    lines = []
    lines.append(f"Parallel analysis: k=95%={k_95}  (showing {n_show_rcs} RCs)")
    lines.append(f"k_95 explains {cumulative_var[k_95-1]:.1%} of total variance")
    lines.append("")
    lines.append("Top 5 loadings per rotated component:")
    lines.append("")
    row_fmt = "  {:<45} {:>+8.3f}  {:>6.2f}  {:>6.2f}  {:>7.3f}"
    hdr = f"  {'Attribute':<45} {'Loading':>8}  {'Mean':>6}  {'Std':>6}  {'Entropy':>7}"
    for s in selected:
        rank = s['rank']
        if rank == k_95 + 1:
            lines.append(f"{'--- below 95% parallel analysis threshold ':─<100}")
        lines.append(f"RC{rank} (representative: {s['attribute']}):")
        lines.append(hdr)
        for name, loading in s['top5']:
            idx = name_to_idx[name]
            lines.append(row_fmt.format(name, loading, col_means[idx], col_stds[idx], col_entropies[idx]))
        lines.append("")

    output = "\n".join(lines)
    print(output)

    txt_path = npy_path.with_name(npy_path.stem + '_rcs.txt')
    txt_path.write_text(output)
    print(f"Saved {txt_path}")

    # Scree plot
    n_show_scree = min(60, num_principles)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_show_scree + 1), eigenvalues[:n_show_scree], 'bo-', label='Data')
    plt.plot(range(1, n_show_scree + 1), threshold_95[:n_show_scree], 'r--', label='95th percentile (random)')
    plt.xlabel('Component')
    plt.ylabel('Eigenvalue')
    plt.title(f'Scree Plot with Parallel Analysis (k_95={k_95})')
    plt.axvline(x=k_95, color='red', linestyle=':', alpha=0.5, label=f'k_95={k_95}')
    plt.legend()
    plt.tight_layout()
    scree_path = npy_path.with_name(npy_path.stem + '_scree.png')
    plt.savefig(scree_path, dpi=150)
    plt.close()
    print(f"Saved {scree_path}")

    # =========================================================================
    # 4. UMAP
    # =========================================================================
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
    umap_path = npy_path.with_name(npy_path.stem + '_umap.png')
    plt.savefig(umap_path, dpi=200)
    plt.close()
    print(f"Saved {umap_path}")


if __name__ == "__main__":
    main()
