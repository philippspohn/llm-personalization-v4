import hydra
from omegaconf import DictConfig
from candidate_principles import candidate_principles
from hydra.utils import get_original_cwd
import numpy as np
from pathlib import Path
import json


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    npy_path = Path(get_original_cwd()) / cfg.output_path
    scores_matrix = np.load(npy_path)
    principle_names = list(candidate_principles)

    scores_std = (scores_matrix - scores_matrix.mean(axis=0)) / (scores_matrix.std(axis=0) + 1e-8)
    corr_matrix = np.corrcoef(scores_std, rowvar=False)
    abs_corr = np.abs(corr_matrix)
    name_to_idx = {n: i for i, n in enumerate(principle_names)}

    json_files = sorted(npy_path.parent.glob(npy_path.stem + '_groups_*.json'))
    if not json_files:
        print("No *_groups_*.json files found. Run the extraction scripts first.")
        return

    results = []
    for jf in json_files:
        data = json.loads(jf.read_text())
        groups = data['groups']

        rep_indices = [name_to_idx[g['representative']] for g in groups if g['representative'] in name_to_idx]
        k = len(rep_indices)

        # Mean pairwise |corr| among representatives
        if k > 1:
            rep_corr_vals = [abs_corr[rep_indices[i], rep_indices[j]]
                             for i in range(k) for j in range(i + 1, k)]
            rep_corr = float(np.mean(rep_corr_vals))
        else:
            rep_corr = float('nan')

        # Coverage: for each principle, max |corr| to any representative; averaged over all principles
        rep_mat = abs_corr[np.ix_(rep_indices, np.arange(len(principle_names)))]  # (k, n)
        coverage = float(rep_mat.max(axis=0).mean())

        # Per-group intra / inter / size
        intras, inters, sizes = [], [], []
        all_indices = np.arange(len(principle_names))
        for g in groups:
            indices = np.array([name_to_idx[m] for m in g['members'] if m in name_to_idx])
            sizes.append(len(indices))
            if len(indices) > 1:
                sub = abs_corr[np.ix_(indices, indices)].copy()
                np.fill_diagonal(sub, np.nan)
                intras.append(float(np.nanmean(sub)))
            other = np.setdiff1d(all_indices, indices)
            if len(other):
                inters.append(float(abs_corr[np.ix_(indices, other)].mean()))

        sizes = np.array(sizes)
        results.append({
            'file': jf.name,
            'method': data['method'],
            'setting': data['setting'],
            'k': k,
            'representatives': [g['representative'] for g in groups],
            'rep_corr': rep_corr,
            'coverage': coverage,
            'intra_mean': float(np.mean(intras)) if intras else float('nan'),
            'intra_std': float(np.std(intras)) if intras else float('nan'),
            'inter_mean': float(np.mean(inters)) if inters else float('nan'),
            'size_min': int(sizes.min()),
            'size_max': int(sizes.max()),
            'size_mean': float(sizes.mean()),
            'size_std': float(sizes.std()),
        })

    results.sort(key=lambda r: (r['method'], r['k']))

    fmt = "{:<35s} {:>4}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}"
    header = fmt.format('setting', 'k', 'rep_corr', 'coverage', 'intra_mn', 'intra_sd', 'inter_mn', 'sz_mean', 'sz_min', 'sz_max')
    print(header)
    print('-' * len(header))
    for r in results:
        label = f"{r['method']}  {r['setting']}"
        print(fmt.format(
            label, r['k'],
            f"{r['rep_corr']:.3f}" if not np.isnan(r['rep_corr']) else '   —',
            f"{r['coverage']:.3f}",
            f"{r['intra_mean']:.3f}" if not np.isnan(r['intra_mean']) else '   —',
            f"{r['intra_std']:.3f}" if not np.isnan(r['intra_std']) else '   —',
            f"{r['inter_mean']:.3f}" if not np.isnan(r['inter_mean']) else '   —',
            f"{r['size_mean']:.1f}",
            str(r['size_min']),
            str(r['size_max']),
        ))

    # Representatives with top alternatives
    # Re-load groups to get members
    jf_map = {jf.name: jf for jf in json_files}
    print()
    for r in results:
        print(f"{r['method']}  {r['setting']}  (k={r['k']}):")
        data = json.loads(jf_map[r['file']].read_text())
        for g in data['groups']:
            alts = [m for m in g['members'] if m != g['representative']][:5]
            alts_str = ', '.join(alts) if alts else '—'
            print(f"  {g['representative']:<40s}")
            # print(f"  {g['representative']:<40s}  [{alts_str}]")
        print()


if __name__ == "__main__":
    main()
