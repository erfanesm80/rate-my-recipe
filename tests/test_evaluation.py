import pandas as pd
import pytest

from rate_my_recipe.evaluation import RankingEvaluator
from rate_my_recipe.ranking import RecipeRanker


@pytest.fixture
def sample_evaluator() -> RankingEvaluator:
    naive_ranking = pd.DataFrame(
        {
            "recipe_id": [101, 102, 103],
            "rank": [1, 2, 3],
            "ranking_score": [5.0, 4.8, 4.0],
            "valid_rating_count": [1, 100, 20],
        }
    )

    bayesian_ranking = pd.DataFrame(
        {
            "recipe_id": [102, 103, 101],
            "rank": [1, 2, 3],
            "ranking_score": [4.7, 4.2, 4.1],
            "valid_rating_count": [100, 20, 1],
        }
    )

    return RankingEvaluator(naive_ranking, bayesian_ranking)


@pytest.fixture
def sample_ranker() -> RecipeRanker:
    recipe_ratings = pd.DataFrame(
        {
            "recipe_id": [101, 102, 103],
            "valid_rating_count": [1, 100, 100],
            "mean_valid_rating": [5.0, 4.8, 3.0],
        }
    )

    return RecipeRanker(recipe_ratings)


def test_top_k_overlap(
    sample_evaluator: RankingEvaluator,
) -> None:
    comparison = sample_evaluator.compare_top_k(top_k=2)

    assert comparison["shared_recipe_count"] == 1
    assert comparison["overlap_percentage"] == 50.0


def test_rank_improvements(
    sample_evaluator: RankingEvaluator,
) -> None:
    rank_changes = sample_evaluator.compare_rank_changes()
    improvements = rank_changes.set_index("recipe_id")["rank_improvement"].to_dict()

    assert improvements == {101: -2, 102: 1, 103: 1}


def test_top_k_rating_counts(
    sample_evaluator: RankingEvaluator,
) -> None:
    comparison = sample_evaluator.compare_top_k_rating_counts(top_k=2).set_index(
        "ranking_method"
    )

    assert comparison.loc["naive", "median_rating_count"] == 50.5
    assert comparison.loc["bayesian", "median_rating_count"] == 60.0


@pytest.mark.parametrize("top_k", [0, -1, 4, 1.5, True])
def test_invalid_top_k(
    sample_evaluator: RankingEvaluator,
    top_k: float,
) -> None:
    with pytest.raises(ValueError, match="top_k"):
        sample_evaluator.compare_top_k(top_k)


def test_different_recipe_populations_are_rejected(
    sample_evaluator: RankingEvaluator,
) -> None:
    invalid_ranking = sample_evaluator.bayesian_ranking.copy()
    invalid_ranking.loc[0, "recipe_id"] = 999

    with pytest.raises(ValueError, match="same recipe IDs"):
        RankingEvaluator(
            sample_evaluator.naive_ranking,
            invalid_ranking,
        )


def test_invalid_ranks_are_rejected(
    sample_evaluator: RankingEvaluator,
) -> None:
    invalid_ranking = sample_evaluator.bayesian_ranking.copy()
    invalid_ranking.loc[0, "rank"] = 2

    with pytest.raises(ValueError, match="consecutive"):
        RankingEvaluator(
            sample_evaluator.naive_ranking,
            invalid_ranking,
        )


def test_missing_columns_are_rejected(
    sample_evaluator: RankingEvaluator,
) -> None:
    invalid_ranking = sample_evaluator.bayesian_ranking.drop(columns="ranking_score")

    with pytest.raises(ValueError, match="Missing ranking columns"):
        RankingEvaluator(
            sample_evaluator.naive_ranking,
            invalid_ranking,
        )


def test_smoothing_sensitivity(
    sample_ranker: RecipeRanker,
) -> None:
    evaluator = RankingEvaluator(
        sample_ranker.rank_by_mean_rating(),
        sample_ranker.rank_by_bayesian_score(10),
    )

    results = evaluator.evaluate_smoothing_sensitivity(
        recipe_ranker=sample_ranker,
        smoothing_strengths=[0.1, 1, 10],
        top_k=1,
    )

    assert results["smoothing_strength"].tolist() == [0.1, 1, 10]
    assert results["overlap_with_naive_percentage"].tolist() == [100.0, 0.0, 0.0]
    assert results["median_top_k_rating_count"].tolist() == [1.0, 100.0, 100.0]


def test_empty_smoothing_strengths_are_rejected(
    sample_ranker: RecipeRanker,
) -> None:
    evaluator = RankingEvaluator(
        sample_ranker.rank_by_mean_rating(),
        sample_ranker.rank_by_bayesian_score(10),
    )

    with pytest.raises(ValueError, match="at least one"):
        evaluator.evaluate_smoothing_sensitivity(
            recipe_ranker=sample_ranker,
            smoothing_strengths=[],
            top_k=1,
        )
