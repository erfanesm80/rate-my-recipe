import pandas as pd
import pytest

from rate_my_recipe.ranking import RecipeRanker


def test_global_mean_uses_rating_counts(sample_ranker: RecipeRanker) -> None:
    expected_mean = (5.0 + 100 * 4.8 + 20 * 4.6 + 100 * 3.0) / 221

    assert sample_ranker.calculate_global_mean_rating() == pytest.approx(expected_mean)


def test_naive_ranking(sample_ranker: RecipeRanker) -> None:
    ranking = sample_ranker.rank_by_mean_rating()

    assert ranking["recipe_id"].tolist() == [101, 102, 103, 104]
    assert ranking["ranking_score"].tolist() == [5.0, 4.8, 4.6, 3.0]
    assert ranking["rank"].tolist() == [1, 2, 3, 4]


def test_bayesian_scoring(sample_ranker: RecipeRanker) -> None:
    global_mean = (5.0 + 100 * 4.8 + 20 * 4.6 + 100 * 3.0) / 221
    smoothing_strength = 10
    ranking = sample_ranker.rank_by_bayesian_score(smoothing_strength)

    scores = ranking.set_index("recipe_id")["ranking_score"].to_dict()
    expected_scores = {
        101: (5.0 + 10 * global_mean) / 11,
        102: (480.0 + 10 * global_mean) / 110,
        103: (92.0 + 10 * global_mean) / 30,
        104: (300.0 + 10 * global_mean) / 110,
    }

    assert scores == pytest.approx(expected_scores)
    assert ranking["recipe_id"].tolist() == [102, 103, 101, 104]
    assert ranking["rank"].tolist() == [1, 2, 3, 4]


def test_bayesian_ranking_can_reverse_naive_order(
    sample_ranker: RecipeRanker,
) -> None:
    assert sample_ranker.rank_by_mean_rating().iloc[0]["recipe_id"] == 101
    assert sample_ranker.rank_by_bayesian_score(10).iloc[0]["recipe_id"] == 102


def test_ranking_ties_use_count_then_recipe_id() -> None:
    recipe_ratings = pd.DataFrame(
        {
            "recipe_id": [103, 102, 101],
            "valid_rating_count": [2, 3, 3],
            "mean_valid_rating": [5.0, 5.0, 5.0],
        }
    )

    ranker = RecipeRanker(recipe_ratings)
    expected_order = [101, 102, 103]

    assert ranker.rank_by_mean_rating()["recipe_id"].tolist() == expected_order
    assert ranker.rank_by_bayesian_score(10)["recipe_id"].tolist() == expected_order


@pytest.mark.parametrize(
    "smoothing_strength",
    [0, -1, float("nan"), float("inf"), True],
)
def test_invalid_smoothing_strength(
    sample_ranker: RecipeRanker,
    smoothing_strength: float | bool,
) -> None:
    with pytest.raises(ValueError, match="Smoothing strength"):
        sample_ranker.rank_by_bayesian_score(smoothing_strength)


def test_ranking_without_valid_ratings(sample_ranker: RecipeRanker) -> None:
    summary = sample_ranker.recipe_rating_summary
    ranker = RecipeRanker(summary.loc[summary["valid_rating_count"].eq(0)])

    assert ranker.rank_by_mean_rating().empty

    with pytest.raises(ValueError, match="without valid ratings"):
        ranker.rank_by_bayesian_score()


@pytest.mark.parametrize(
    "rating_count,mean_rating",
    [
        (-1, 4.0),
        (1.5, 4.0),
        (0, 5.0),
        (2, 6.0),
        (2, float("nan")),
    ],
)
def test_invalid_recipe_statistics(
    rating_count: float,
    mean_rating: float,
) -> None:
    recipe_ratings = pd.DataFrame(
        {
            "recipe_id": [101],
            "valid_rating_count": [rating_count],
            "mean_valid_rating": [mean_rating],
        }
    )

    with pytest.raises(ValueError):
        RecipeRanker(recipe_ratings)


def test_duplicate_recipe_ids_are_rejected(sample_ranker: RecipeRanker) -> None:
    duplicate_recipes = pd.concat(
        [
            sample_ranker.recipe_rating_summary,
            sample_ranker.recipe_rating_summary.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="unique"):
        RecipeRanker(duplicate_recipes)


def test_missing_required_columns_are_rejected(
    sample_ranker: RecipeRanker,
) -> None:
    incomplete_recipes = sample_ranker.recipe_rating_summary.drop(
        columns="valid_rating_count"
    )

    with pytest.raises(ValueError, match="Missing ranking columns"):
        RecipeRanker(incomplete_recipes)
