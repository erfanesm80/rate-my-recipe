"""Interactive dashboard for comparing Food.com recipe rankings.

Run from the repository root with ``streamlit run app.py`` after generating
``data/processed/recipe_rating_summary.csv`` in the exploration notebook.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from rate_my_recipe.evaluation import RankingEvaluator
from rate_my_recipe.ranking import RecipeRanker

PROJECT_ROOT = Path(__file__).resolve().parent
RATING_SUMMARY_CSV = PROJECT_ROOT / "data" / "processed" / "recipe_rating_summary.csv"
TOP_K_OPTIONS = [10, 20, 50, 100]

DISPLAY_COLUMNS = {
    "rank": "Global rank",
    "recipe_name": "Recipe",
    "valid_rating_count": "Valid ratings",
    "mean_valid_rating": "Mean rating",
    "ranking_score": "Ranking score",
    "preparation_time_minutes": "Minutes",
    "ingredient_count": "Ingredients",
}


@st.cache_data(show_spinner="Loading the prepared recipe data…", max_entries=2)
def load_recipe_summary(csv_path: str, modified_time_ns: int) -> pd.DataFrame:
    """Read the prepared data, invalidating the cache when the CSV changes."""
    return pd.read_csv(csv_path)


def select_leaderboard(
    ranking: pd.DataFrame,
    minimum_ratings: int,
    search_text: str,
    top_k: int,
) -> pd.DataFrame:
    """Filter an existing global ranking without modifying scores or ranks."""
    matches = ranking["valid_rating_count"].ge(minimum_ratings)
    if search_text.strip():
        matches &= ranking["recipe_name"].str.contains(
            search_text.strip(), case=False, regex=False, na=False
        )
    return ranking.loc[matches].head(top_k).copy()


def format_leaderboard(ranking: pd.DataFrame) -> pd.DataFrame:
    """Select and rename the columns shown in the table and CSV export."""
    return ranking[list(DISPLAY_COLUMNS)].rename(columns=DISPLAY_COLUMNS)


def show_leaderboard(ranking: pd.DataFrame, compact: bool = False) -> None:
    """Display a leaderboard with consistent numeric formatting."""
    table = format_leaderboard(ranking)
    if compact:
        table = table[["Global rank", "Recipe", "Valid ratings", "Ranking score"]]

    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "Mean rating": st.column_config.NumberColumn(format="%.3f"),
            "Ranking score": st.column_config.NumberColumn(format="%.3f"),
            "Minutes": st.column_config.NumberColumn(format="%d"),
        },
    )


def main() -> None:
    """Render the dashboard, using the same ranking logic as the notebooks."""
    st.set_page_config(page_title="Rate My Recipe", page_icon="🍽️", layout="wide")
    st.title("🍽️ Rate My Recipe")
    st.write(
        "Compare rankings based on average star ratings and rating counts. "
        "Scores are computed from the full Food.com dataset."
    )

    if not RATING_SUMMARY_CSV.is_file():
        st.error("Prepared recipe data not found.")
        st.info(
            "Download the Food.com CSVs as described in the README, then run "
            "`notebooks/data_exploration.ipynb` to create the processed summary."
        )
        st.code("data/processed/recipe_rating_summary.csv")
        st.stop()

    try:
        recipe_summary = load_recipe_summary(
            str(RATING_SUMMARY_CSV), RATING_SUMMARY_CSV.stat().st_mtime_ns
        )
        recipe_ranker = RecipeRanker(recipe_summary)
        naive_ranking = recipe_ranker.rank_by_mean_rating()
        global_mean = recipe_ranker.calculate_global_mean_rating()
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as error:
        st.error(f"Could not load the prepared recipe data: {error}")
        st.stop()

    st.sidebar.header("Ranking settings")
    smoothing_strength = st.sidebar.slider(
        "Smoothing strength (m)",
        min_value=1,
        max_value=100,
        value=10,
        help="Higher values put more weight on the global mean rating.",
    )
    top_k = st.sidebar.selectbox("Recipes to show (top-k)", TOP_K_OPTIONS, index=1)
    ranking_method = st.sidebar.radio(
        "Leaderboard method", ["Weighted score", "Naive mean"], index=0
    )
    st.sidebar.divider()
    minimum_ratings = st.sidebar.number_input(
        "Minimum valid ratings", min_value=1, value=1, step=1
    )
    search_text = st.sidebar.text_input("Find recipes by name", placeholder="e.g. soup")
    st.sidebar.caption(
        "Filters change which recipes are displayed, not the global scores or ranks."
    )

    weighted_ranking = recipe_ranker.rank_by_bayesian_score(float(smoothing_strength))
    evaluator = RankingEvaluator(naive_ranking, weighted_ranking)

    col_recipes, col_ratings, col_mean = st.columns(3)
    col_recipes.metric("Rated recipes", f"{len(naive_ranking):,}")
    col_ratings.metric(
        "Valid star ratings", f"{int(recipe_summary['valid_rating_count'].sum()):,}"
    )
    col_mean.metric("Global mean (C)", f"{global_mean:.4f} / 5")

    leaderboard_tab, comparison_tab = st.tabs(["Leaderboard", "Compare methods"])

    with leaderboard_tab:
        ranking = (
            weighted_ranking if ranking_method == "Weighted score" else naive_ranking
        )
        displayed_ranking = select_leaderboard(
            ranking, minimum_ratings, search_text, top_k
        )
        st.subheader(f"{ranking_method} leaderboard")
        st.caption(
            "Ranks are global positions. Filtering does not recalculate scores "
            "or restart rank numbering."
        )
        if displayed_ranking.empty:
            st.warning("No recipes match these filters. Try a different search.")
        else:
            show_leaderboard(displayed_ranking)
            st.download_button(
                "Download displayed rows (CSV)",
                data=format_leaderboard(displayed_ranking)
                .to_csv(index=False)
                .encode("utf-8"),
                file_name=f"recipe_leaderboard_{ranking_method.lower().replace(' ', '_')}.csv",
                mime="text/csv",
            )

    with comparison_tab:
        st.subheader(f"Global ranking comparison · m = {smoothing_strength}")
        st.caption(
            "Comparisons use the complete, unfiltered leaderboards. "
            "The name and minimum-rating filters do not affect these statistics."
        )
        overlap = evaluator.compare_top_k(top_k)
        rating_counts = evaluator.compare_top_k_rating_counts(top_k)
        median_by_method = rating_counts.set_index("ranking_method")[
            "median_rating_count"
        ]

        naive_median = median_by_method["naive"]
        weighted_median = median_by_method["bayesian"]

        overlap_col, naive_col, weighted_col = st.columns(3)
        overlap_col.metric(
            f"Top-{top_k} overlap", f"{overlap['overlap_percentage']:.0f}%"
        )
        naive_col.metric("Naive median ratings", f"{naive_median:g}")
        weighted_col.metric("Weighted median ratings", f"{weighted_median:g}")

        overlap_table = pd.DataFrame(
            [evaluator.compare_top_k(k) for k in TOP_K_OPTIONS]
        )
        chart_col, evidence_col = st.columns(2)
        with chart_col:
            st.markdown("**Overlap at different cutoffs**")
            st.line_chart(
                overlap_table,
                x="top_k",
                y="overlap_percentage",
                x_label="Top-k",
                y_label="Shared recipes (%)",
                height=300,
            )
        with evidence_col:
            st.markdown(f"**Median valid ratings in the top {top_k}**")
            st.bar_chart(
                rating_counts,
                x="ranking_method",
                y="median_rating_count",
                x_label="Ranking method",
                y_label="Median valid ratings",
                height=300,
            )

        st.markdown("**Top-10 preview from each method**")
        left, right = st.columns(2)
        with left:
            st.caption("Naive mean")
            show_leaderboard(naive_ranking.head(10), compact=True)
        with right:
            st.caption(f"Weighted score (m = {smoothing_strength})")
            show_leaderboard(weighted_ranking.head(10), compact=True)

    st.divider()
    st.caption(
        "Bayesian-style shrinkage is a count-aware ranking heuristic, not a "
        "validated predictor of future satisfaction. For detailed results "
        "and limitations, see RESULTS.md."
    )


if __name__ == "__main__":
    main()
