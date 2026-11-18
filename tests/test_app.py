"""Checks for Streamlit's filtering and table-formatting helpers."""

import pandas as pd

from app import format_leaderboard, select_leaderboard


def sample_ranking() -> pd.DataFrame:
    """Provide a small, already-ranked set of recipes."""
    return pd.DataFrame(
        {
            "rank": [1, 2, 3],
            "recipe_name": ["Tomato Soup", "Banana Bread", "Spicy SOUP"],
            "valid_rating_count": [2, 20, 10],
            "mean_valid_rating": [5.0, 4.9, 4.8],
            "ranking_score": [4.9, 4.85, 4.8],
            "preparation_time_minutes": [20, 60, 30],
            "ingredient_count": [5, 8, 4],
        }
    )


def test_filters_preserve_global_ranks() -> None:
    result = select_leaderboard(sample_ranking(), 10, "", 20)

    assert result["rank"].tolist() == [2, 3]
    assert result["recipe_name"].tolist() == ["Banana Bread", "Spicy SOUP"]


def test_name_search_is_case_insensitive_and_literal() -> None:
    result = select_leaderboard(sample_ranking(), 1, " soup ", 10)
    special_character_search = select_leaderboard(sample_ranking(), 1, ".*", 10)

    assert result["rank"].tolist() == [1, 3]
    assert special_character_search.empty


def test_name_search_handles_missing_names() -> None:
    ranking = sample_ranking()
    ranking.loc[1, "recipe_name"] = None

    result = select_leaderboard(ranking, 1, "soup", 10)

    assert result["rank"].tolist() == [1, 3]


def test_top_k_applies_after_filtering() -> None:
    result = select_leaderboard(
        sample_ranking(),
        minimum_ratings=10,
        search_text="",
        top_k=1,
    )

    assert result["rank"].tolist() == [2]


def test_formatting_keeps_scores_without_changing_values() -> None:
    result = format_leaderboard(sample_ranking().head(1))

    assert result.loc[0, "Global rank"] == 1
    assert result.loc[0, "Ranking score"] == 4.9
    assert result.loc[0, "Valid ratings"] == 2
