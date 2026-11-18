# Rate My Recipe

**Ranking Food.com recipes using star ratings and rating counts**

An individual Python project for *Coding for Data Science and Data Management* at the University of Milan (academic year 2026/27).

## Overview

A recipe with a five-star average from one user should not necessarily outrank a recipe with a slightly lower average supported by hundreds of ratings. Ranking solely by mean rating ignores how much evidence supports each score.

**Rate My Recipe** compares two ways of ranking Food.com recipes:

1. **Naive ranking:** sort recipes by their average valid star rating.
2. **Bayesian-style weighted ranking:** shrink each recipe's average toward the global rating mean, with the amount of shrinkage depending on its number of valid ratings.

The project explores the dataset, prepares validated recipe-level statistics, compares the rankings numerically and visually, and examines sensitivity to the smoothing parameter. The goal is to understand how accounting for rating counts changes the leaderboard—not to claim that either method predicts future satisfaction better.

## Project Structure

```text
rate-my-recipe/
├── README.md
├── RESULTS.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── ranking_analysis.ipynb
│   └── project_presentation.ipynb
├── rate_my_recipe/
│   ├── __init__.py
│   ├── data_preparation.py
│   ├── ranking.py
│   └── evaluation.py
├── tests/
│   ├── conftest.py
│   ├── test_data_preparation.py
│   ├── test_ranking.py
│   └── test_evaluation.py
└── data/                         # Local only; ignored by Git
    ├── raw/
    │   ├── RAW_recipes.csv
    │   └── RAW_interactions.csv
    └── processed/
        ├── recipe_rating_summary.csv
        └── recipe_tag_index.csv
```

The `data/` directory and its CSV files are created locally and are **not tracked in Git**.

### Reusable Python Modules

| Module | Responsibility |
|---|---|
| `data_preparation.py` | Validate raw data, aggregate valid ratings, and normalize recipe tags |
| `ranking.py` | Compute naive and Bayesian-style scores and deterministic rankings |
| `evaluation.py` | Compare top-k lists, rank movements, rating counts, and smoothing sensitivity |

The notebooks import these modules instead of duplicating the data-preparation or ranking algorithms.

## Dataset

**Source:** [Food.com Recipes and Interactions (Kaggle)](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)

Download and extract the dataset, then place these files in `data/raw/`:

- **`RAW_recipes.csv`** — recipe IDs, names, preparation times, ingredient counts, tags, and other metadata.
- **`RAW_interactions.csv`** — user IDs, recipe IDs, dates, star ratings, and text reviews.

The dataset requires no external API during analysis. It is excluded from the repository because of its size.

## Installation

The project was developed with **Python 3.13.15**. Install the versions listed in `requirements.txt` for reproducibility.

```bash
git clone https://github.com/erfanesm80/rate-my-recipe.git
cd rate-my-recipe

python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

These environment commands are for macOS/Linux. An existing compatible Python environment, including Poetry, can also be used.

Create the local data directories and copy the downloaded CSVs into `data/raw/`:

```bash
mkdir -p data/raw data/processed
```

## Running the Notebooks

From the repository root, start JupyterLab:

```bash
jupyter lab notebooks/
```

Open and run the notebooks in the following order. Use **Restart Kernel and Run All Cells** to reproduce the results.

| Notebook | Purpose | Input / output |
|---|---|---|
| **[1. Data Exploration and Preparation](notebooks/data_exploration.ipynb)** | Explore schemas, missing values, rating distributions, and data-quality issues; validate and aggregate ratings; normalize tags | Reads both raw CSVs; exports `recipe_rating_summary.csv` and `recipe_tag_index.csv` |
| **[2. Ranking and Statistical Evaluation](notebooks/ranking_analysis.ipynb)** | Compare the naive and count-aware rankings, inspect top-k overlap and rank movements, evaluate rating-count differences and smoothing sensitivity | Reads `recipe_rating_summary.csv`; displays tables and plots |
| **[3. Project Presentation](notebooks/project_presentation.ipynb)** | Present the research question, method, main empirical findings, and limitations in a concise format | Reads `recipe_rating_summary.csv`; displays presentation-ready results |

**Dependency:** Notebook 1 must be run first to generate the processed rating summary. Notebooks 2 and 3 can then be executed **independently**; the presentation does not require Notebook 2 to have run.

Saved notebook outputs can be viewed on GitHub, but reproducing the results locally requires the original CSVs.

## Methodology

### Rating Preparation

The original interaction data contain ratings between **0 and 5**. For this project:

- Ratings **1–5** are valid star feedback and are used for scoring.
- A rating of **0** is treated as an interaction without usable star feedback. It counts toward total interactions but not toward valid-rating counts or averages.
- Recipes without valid star ratings remain in the prepared dataset but are **excluded from both ranked lists**.

The prepared `recipe_rating_summary.csv` contains one row per recipe:

| Column | Meaning |
|---|---|
| `recipe_id` | Unique recipe identifier |
| `recipe_name` | Recipe name |
| `preparation_time_minutes` | Reported preparation time |
| `ingredient_count` | Number of ingredients |
| `interaction_count` | All interactions, including zero ratings |
| `valid_rating_count` | Number of ratings from 1 to 5 |
| `mean_valid_rating` | Average of the valid ratings; missing for unrated recipes |

The optional `recipe_tag_index.csv` contains unique, normalized `(recipe_id, tag)` pairs. Tags and other recipe attributes describe recipes but do not influence the **global** rankings used in this project.

### Naive Mean-Rating Baseline

For recipe $i$, the baseline score is the observed average of its valid ratings:

$$
S_i^{\mathrm{naive}} = \bar r_i
$$

This score does not distinguish between an average based on one rating and the same average supported by many ratings.

### Bayesian-Style Weighted Score

The count-aware method uses:

$$
S_i^{\mathrm{weighted}} = \frac{n_i\bar r_i + mC}{n_i + m}
$$

where:

- $\bar r_i$ is recipe $i$'s average valid star rating;
- $n_i$ is its number of valid ratings;
- $C$ is the **global mean of individual valid ratings**, equivalently $\sum_i n_i\bar r_i / \sum_i n_i$;
- $m > 0$ is the smoothing strength, measured in rating-equivalent weight.

When $n_i$ is small, the score is pulled more strongly toward $C$. As $n_i$ grows, the observed mean receives more weight. The main comparison uses **$m=10$** as an illustrative choice; additional values are assessed through sensitivity analysis. This is a Bayesian-style shrinkage heuristic, **not** a claim of an optimally fitted probabilistic model.

Both ranking methods use deterministic tie-breaking: **score descending, valid-rating count descending, recipe ID ascending**.

### Evaluation

The analysis investigates:

- **Top-k overlap:** the share of recipes appearing in both top-k lists.
- **Rank changes:** each recipe's naive rank minus its weighted rank; positive values indicate an improvement under the weighted method.
- **Rating-count evidence:** the distribution of valid-rating counts among top-ranked recipes.
- **Smoothing sensitivity:** how changes in $m$ affect top-k membership and rating-count evidence.

Numerical findings and visualizations appear in the [ranking analysis notebook](notebooks/ranking_analysis.ipynb). The [presentation notebook](notebooks/project_presentation.ipynb) summarizes the principal results for the oral discussion.

## Tests and Code Quality

Run the test suite from the repository root:

```bash
python -m pytest tests/ -v
```

Check the Python modules and tests with Ruff:

```bash
ruff check rate_my_recipe tests
```

The tests use small synthetic datasets, so **the original Food.com CSVs are not needed to run them**. They cover input validation, zero and invalid ratings, aggregation, tag parsing, ranking formulas, deterministic ties, evaluation calculations, and error conditions. Shared test fixtures are defined in `tests/conftest.py`.

## Results and Limitations

See **[RESULTS.md](RESULTS.md)** for the key numerical findings, their interpretation, and the limitations of this analysis. The notebooks contain the complete calculations and visualizations.

## References

- [Food.com Recipes and Interactions — Kaggle](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)
- [Evan Miller (2009), *How Not to Sort by Average Rating*](https://www.evanmiller.org/how-not-to-sort-by-average-rating.html) — motivates why rating counts matter. The Wilson lower bound discussed there applies to binary outcomes and is **not** applied directly to the 1–5-star ratings in this project.
- [Coding for Data Science and Data Management — University of Milan](https://github.com/sergiopicascia/codingdse-py)
