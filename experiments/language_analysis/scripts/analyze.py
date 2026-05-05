"""Compare word/n-gram usage between subsets of the synthetic dataset.

Sample = first user prompt of each conversation. For each user with attribute
A and side S (follow|avoid), every conversation contributes one sample tagged
(A, S).

Two metrics per n-gram per comparison:
  * doc_freq_diff: share of group-A samples containing the n-gram minus the
    same share in group B. Interpretable, biased toward common terms.
  * log_odds_z: log-odds ratio with informative Dirichlet prior (Monroe,
    Colaresi & Quinn 2008, "Fightin' Words"). z>1.96 ~ significant.

Comparisons:
  * follow_vs_avoid: for each attribute, its follow-side samples vs its
    avoid-side samples.
  * attribute_vs_others: for each attribute, its samples vs the union of
    all other attributes' samples (restricted to a chosen side, default
    follow — i.e. "what distinguishes prompts from users with attribute X
    set to follow").
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

REPO_ROOT = Path(__file__).resolve().parents[3]

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str, min_len: int) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text) if len(t) >= min_len]


def ngrams(tokens: list[str], n: int) -> list[str]:
    if n == 1:
        return tokens
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def load_samples(path: Path) -> list[dict]:
    """Each returned dict: {attribute, side, user_idx, tokens_ignored, text}."""
    samples = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            attrs = row["rewrite_style_attributes"]
            if len(attrs) != 1:
                raise ValueError(
                    f"expected 1 attribute per user, got {len(attrs)} "
                    f"(user_idx={row['user_idx']})"
                )
            attribute = attrs[0]["attribute"]
            side = attrs[0]["side"]
            for conv in row["conversations"]:
                first_user = next(
                    (m for m in conv["messages"] if m.get("role") == "user"),
                    None,
                )
                if first_user is None:
                    continue
                samples.append(
                    {
                        "attribute": attribute,
                        "side": side,
                        "user_idx": row["user_idx"],
                        "text": first_user["content"],
                    }
                )
    return samples


def build_ngram_index(
    samples: list[dict], n: int, min_token_len: int, min_doc_freq: int
) -> tuple[list[set[str]], list[str]]:
    """Returns (per_sample_ngram_sets, vocab) after pruning rare n-grams."""
    sample_sets: list[set[str]] = []
    df: Counter[str] = Counter()
    for s in samples:
        toks = tokenize(s["text"], min_token_len)
        gset = set(ngrams(toks, n))
        sample_sets.append(gset)
        for g in gset:
            df[g] += 1
    vocab = [g for g, c in df.items() if c >= min_doc_freq]
    vocab_set = set(vocab)
    sample_sets = [gs & vocab_set for gs in sample_sets]
    return sample_sets, vocab


def doc_freq(sets: list[set[str]], idx: list[int], vocab: list[str]) -> Counter:
    c: Counter[str] = Counter()
    for i in idx:
        for g in sets[i]:
            c[g] += 1
    return c


def log_odds_dirichlet(
    df_a: Counter, df_b: Counter, n_a: int, n_b: int, vocab: list[str], alpha: float
) -> dict[str, float]:
    """Per-sample document-frequency log-odds with informative Dirichlet prior.

    Standard Monroe et al. formulation but on document counts (number of
    samples containing the n-gram) rather than token counts. Same mechanics:
    z = (log-odds_a - log-odds_b) / sqrt(var). Prior is the pooled distribution
    scaled by alpha * V.
    """
    V = len(vocab)
    # Pooled per-sample freq used as prior.
    pooled = {g: df_a.get(g, 0) + df_b.get(g, 0) for g in vocab}
    total_pooled = sum(pooled.values())
    if total_pooled == 0:
        return {g: 0.0 for g in vocab}
    prior_mass = alpha * V
    a0 = sum(df_a.values()) + prior_mass
    b0 = sum(df_b.values()) + prior_mass
    z = {}
    for g in vocab:
        prior_g = prior_mass * (pooled[g] / total_pooled) if total_pooled else 0.0
        ag = df_a.get(g, 0) + prior_g
        bg = df_b.get(g, 0) + prior_g
        # Guard against zeros — prior_g can be 0 if pooled count is 0 (rare
        # after vocab pruning, but safe).
        if ag <= 0 or bg <= 0 or a0 - ag <= 0 or b0 - bg <= 0:
            z[g] = 0.0
            continue
        log_odds = math.log(ag / (a0 - ag)) - math.log(bg / (b0 - bg))
        var = 1.0 / ag + 1.0 / bg
        z[g] = log_odds / math.sqrt(var)
    return z


def score_comparison(
    sets: list[set[str]],
    vocab: list[str],
    idx_a: list[int],
    idx_b: list[int],
    alpha: float,
) -> pd.DataFrame:
    n_a, n_b = len(idx_a), len(idx_b)
    df_a = doc_freq(sets, idx_a, vocab)
    df_b = doc_freq(sets, idx_b, vocab)
    z = log_odds_dirichlet(df_a, df_b, n_a, n_b, vocab, alpha)
    eps = 0.5  # add-one-half smoothing for the ratio so 0-counts don't blow up
    n_total = n_a + n_b
    rows = []
    for g in vocab:
        ca, cb = df_a.get(g, 0), df_b.get(g, 0)
        fa = ca / n_a if n_a else 0.0
        fb = cb / n_b if n_b else 0.0
        eps_a = eps / n_a if n_a else 0.0
        eps_b = eps / n_b if n_b else 0.0
        # Normalized PMI for "n-gram appears" with "sample is in group A".
        # NPMI = log(p(g, A) / (p(g) p(A))) / -log p(g, A). Bounded in [-1, 1].
        # Surfaces terms that are distinctive *because rare-elsewhere*, which
        # log-odds-z under-rewards.
        p_g = (ca + cb + eps) / (n_total + eps)
        p_a = n_a / n_total if n_total else 0.0
        p_g_a = (ca + eps) / (n_total + eps)
        if p_g > 0 and p_a > 0 and p_g_a > 0:
            pmi = math.log(p_g_a / (p_g * p_a))
            npmi = pmi / (-math.log(p_g_a))
        else:
            npmi = 0.0
        rows.append(
            {
                "ngram": g,
                "count_a": ca,
                "count_b": cb,
                "doc_freq_a": fa,
                "doc_freq_b": fb,
                "doc_freq_diff": fa - fb,
                "freq_ratio": (fa + eps_a) / (fb + eps_b),
                "log_odds_z": z[g],
                "npmi_a": npmi,
            }
        )
    return pd.DataFrame(rows)


def write_top(
    df: pd.DataFrame,
    path: Path,
    top_k: int,
    n_a: int,
    n_b: int,
    min_abs_z: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path.with_suffix(".full.csv"), index=False)
    sig = df[df["log_odds_z"].abs() >= min_abs_z]
    pos = sig.nlargest(top_k, "log_odds_z")
    neg = sig.nsmallest(top_k, "log_odds_z")
    npmi_a = sig.nlargest(top_k, "npmi_a")
    npmi_b = sig.nsmallest(top_k, "npmi_a")
    cols = ["ngram", "count_a", "count_b", "doc_freq_a", "doc_freq_b",
            "freq_ratio", "log_odds_z", "npmi_a"]
    with path.with_suffix(".top.txt").open("w") as f:
        f.write(f"# n_a={n_a}  n_b={n_b}  min_abs_z={min_abs_z}\n")
        f.write(f"# {len(sig)} of {len(df)} n-grams pass the |z| threshold\n\n")
        f.write("## Top by log-odds z (distinctive in A)\n")
        f.write(pos[cols].to_string(index=False, float_format="%.3f") + "\n\n")
        f.write("## Top by log-odds z (distinctive in B)\n")
        f.write(neg[cols].to_string(index=False, float_format="%.3f") + "\n\n")
        f.write("## Top by NPMI (distinctive-because-rare-elsewhere, A side)\n")
        f.write(npmi_a[cols].to_string(index=False, float_format="%.3f") + "\n\n")
        f.write("## Top by NPMI (distinctive-because-rare-elsewhere, B side)\n")
        f.write(npmi_b[cols].to_string(index=False, float_format="%.3f") + "\n")


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    in_path = Path(cfg.input_path)
    if not in_path.is_absolute():
        in_path = REPO_ROOT / in_path
    out_dir = Path(cfg.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {in_path}")
    samples = load_samples(in_path)
    print(f"  {len(samples)} samples (first-user-prompt per conversation)")

    if cfg.get("sample_limit"):
        rng = random.Random(cfg.seed)
        groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        for i, s in enumerate(samples):
            groups[(s["attribute"], s["side"])].append(i)
        target = int(cfg.sample_limit)
        per_group = max(1, target // len(groups))
        keep: list[int] = []
        for idxs in groups.values():
            rng.shuffle(idxs)
            keep.extend(idxs[:per_group])
        rng.shuffle(keep)
        keep = keep[:target]
        samples = [samples[i] for i in keep]
        print(f"  capped to {len(samples)} samples (stratified by attribute,side)")

    attributes = sorted({s["attribute"] for s in samples})
    print(f"  attributes: {attributes}")

    by_attr_side: dict[tuple[str, str], list[int]] = defaultdict(list)
    by_attr: dict[str, list[int]] = defaultdict(list)
    by_side: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        by_attr_side[(s["attribute"], s["side"])].append(i)
        by_attr[s["attribute"]].append(i)
        by_side[s["side"]].append(i)

    for n in cfg.ngram_sizes:
        print(f"\n=== n={n} ===")
        sets, vocab = build_ngram_index(
            samples, n, cfg.min_token_len, cfg.min_doc_freq
        )
        print(f"  vocab after df>={cfg.min_doc_freq}: {len(vocab)}")
        n_dir = out_dir / f"n{n}"

        if cfg.comparisons.follow_vs_avoid:
            for attr in attributes:
                idx_a = by_attr_side[(attr, "follow")]
                idx_b = by_attr_side[(attr, "avoid")]
                if not idx_a or not idx_b:
                    continue
                df = score_comparison(sets, vocab, idx_a, idx_b, cfg.log_odds_alpha)
                write_top(
                    df,
                    n_dir / "follow_vs_avoid" / f"{attr}",
                    cfg.top_k,
                    len(idx_a),
                    len(idx_b),
                    cfg.display_min_abs_z,
                )
            # Also overall, ignoring attribute.
            df = score_comparison(
                sets, vocab, by_side["follow"], by_side["avoid"], cfg.log_odds_alpha
            )
            write_top(
                df,
                n_dir / "follow_vs_avoid" / "_ALL",
                cfg.top_k,
                len(by_side["follow"]),
                len(by_side["avoid"]),
                cfg.display_min_abs_z,
            )

        if cfg.comparisons.attribute_vs_others:
            sides_cfg = cfg.comparisons.attribute_vs_others_side
            sides = (
                ["follow", "avoid"]
                if sides_cfg in ("both", "follow_and_avoid")
                else [sides_cfg]
            )
            for side in sides:
                for attr in attributes:
                    idx_a = by_attr_side[(attr, side)]
                    idx_b = [
                        i
                        for a in attributes
                        if a != attr
                        for i in by_attr_side[(a, side)]
                    ]
                    if not idx_a or not idx_b:
                        continue
                    df = score_comparison(
                        sets, vocab, idx_a, idx_b, cfg.log_odds_alpha
                    )
                    write_top(
                        df,
                        n_dir / f"attribute_vs_others_{side}" / f"{attr}",
                        cfg.top_k,
                        len(idx_a),
                        len(idx_b),
                        cfg.display_min_abs_z,
                    )

        if cfg.comparisons.get("attribute_vs_general", False):
            for side in ("follow", "avoid"):
                for attr in attributes:
                    idx_a = by_attr_side[(attr, side)]
                    # B = all prompts whose attribute is NOT `attr` (both
                    # sides mixed). Excludes both endpoints of axis `attr`.
                    idx_b = [
                        i
                        for a in attributes
                        if a != attr
                        for s in ("follow", "avoid")
                        for i in by_attr_side[(a, s)]
                    ]
                    if not idx_a or not idx_b:
                        continue
                    df = score_comparison(
                        sets, vocab, idx_a, idx_b, cfg.log_odds_alpha
                    )
                    write_top(
                        df,
                        n_dir / f"attribute_vs_general_{side}" / f"{attr}",
                        cfg.top_k,
                        len(idx_a),
                        len(idx_b),
                        cfg.display_min_abs_z,
                    )

    print(f"\nwrote results under {out_dir}")


if __name__ == "__main__":
    main()
