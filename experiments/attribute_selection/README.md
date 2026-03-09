# Attribute Selection

The goal is to select maximally independent attributes for judging LLM responses.

## Method

We rate a large pool of candidate attributes across many LLM conversations, then apply an entropy pre-filter (threshold ≥ 1.5) to remove attributes with degenerate distributions (e.g., nearly always scored 1). On the surviving attributes we compute a correlation matrix, run PCA, determine the number of meaningful dimensions via parallel analysis (95th percentile threshold), and apply varimax rotation at each target k to obtain interpretable, orthogonal components. Each rotated component represents an independent dimension of LLM response style, from which we select representative attributes.

## Usage
```bash
python experiments/attribute_selection/scripts/generate_ratings.py
python experiments/attribute_selection/scripts/generate_ratings.py --config-name config_user_prompts

python experiments/attribute_selection/scripts/analyze_ratings.py
python experiments/attribute_selection/scripts/analyze_ratings.py --config-name config_user_prompts
```

## Results

The analysis script produces:
- **Score distributions** (`_score_distributions.png`): per-attribute histograms over the full (unfiltered) pool, useful for inspecting which attributes the judge uses well
- **Scree plot** (`_scree.png`): eigenvalues vs. parallel analysis threshold, showing the dimensionality ceiling
- **Rotated components** (`_rcs.txt`): varimax solutions at each k, with per-attribute loadings, means, standard deviations, and entropies
- **UMAP** (`_umap.png`): 2D projection of attributes using distance = 1 − |correlation|

Results are saved in the `data/attribute_selection` directory.