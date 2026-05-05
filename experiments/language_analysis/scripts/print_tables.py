"""Render the two paper-candidate tables from data/language_analysis CSVs.

Table 1 (follow_vs_avoid per attribute):
  cols  : (follow > avoid)  |  (avoid > follow)
  rows  : attributes
  values: top-K n-grams sorted by |z|, shown as `ngram (z, ratio×)`

Table 2 (each side vs the rest of the corpus, excluding all of attribute X):
  cols  : (follow_X vs all-other-attrs)  |  (avoid_X vs all-other-attrs)
  rows  : attributes
  values: top-K n-grams sorted by z desc, shown as `ngram (z, ratio×)`

Both rendered for unigrams (n=1) and bigrams (n=2).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]


def fmt(row, ratio_invert: bool) -> str:
    z = abs(row.log_odds_z)
    r = (1.0 / row.freq_ratio) if ratio_invert else row.freq_ratio
    return f"{row.ngram} (z={z:.0f}, {r:.1f}×)"


def cell(df: pd.DataFrame, top_k: int, side: str) -> str:
    """side='top'  → terms most overrepresented in A (largest +z).
       side='bot'  → terms most overrepresented in B (largest -z); ratio inverted."""
    if side == "top":
        rows = df.nlargest(top_k, "log_odds_z")
        return ", ".join(fmt(r, ratio_invert=False) for r in rows.itertuples())
    rows = df.nsmallest(top_k, "log_odds_z")
    return ", ".join(fmt(r, ratio_invert=True) for r in rows.itertuples())


def render_table_1(base: Path, n: int, top_k: int) -> str:
    cmp_dir = base / f"n{n}" / "follow_vs_avoid"
    out = [f"### Table 1 ({n}-grams) — follow vs avoid, per attribute", ""]
    out.append(f"{'attribute':<22}  {'follow > avoid':<70}  {'avoid > follow'}")
    out.append("-" * 160)
    for path in sorted(cmp_dir.glob("*.full.csv")):
        attr = path.stem.replace(".full", "")
        if attr == "_ALL":
            continue
        df = pd.read_csv(path)
        out.append(f"{attr:<22}  {cell(df, top_k, 'top'):<70}")
        out.append(f"{'':<22}  {'':<70}  {cell(df, top_k, 'bot')}")
        out.append("")
    return "\n".join(out)


def render_table_2(base: Path, n: int, top_k: int) -> str:
    follow_dir = base / f"n{n}" / "attribute_vs_general_follow"
    avoid_dir = base / f"n{n}" / "attribute_vs_general_avoid"
    out = [
        f"### Table 2 ({n}-grams) — attribute side vs all other attributes "
        f"(both sides mixed)",
        "",
    ]
    out.append(
        f"{'attribute':<22}  {'follow_X vs general':<70}  {'avoid_X vs general'}"
    )
    out.append("-" * 160)
    attrs = sorted(p.stem.replace(".full", "") for p in follow_dir.glob("*.full.csv"))
    for attr in attrs:
        df_f = pd.read_csv(follow_dir / f"{attr}.full.csv")
        df_a = pd.read_csv(avoid_dir / f"{attr}.full.csv")
        out.append(f"{attr:<22}  {cell(df_f, top_k, 'top'):<70}  "
                   f"{cell(df_a, top_k, 'top')}")
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="data/language_analysis")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--ngrams", type=int, nargs="+", default=[1, 2])
    args = p.parse_args()
    base = Path(args.base)
    if not base.is_absolute():
        base = REPO_ROOT / base
    for n in args.ngrams:
        print(render_table_1(base, n, args.top_k))
        print()
        print(render_table_2(base, n, args.top_k))
        print("\n" + "=" * 160 + "\n")


if __name__ == "__main__":
    main()
