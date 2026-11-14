import pytest

from rate_my_recipe.evaluation import RankingEvaluator
from rate_my_recipe.ranking import RecipeRanker


@pytest.fixture
def sample_evaluator(sample_ranker: RecipeRanker) -> RankingEvaluator:
    return RankingEvaluator(
        sample_ranker.rank_by_mean_rating(),
        sample_ranker.rank_by_bayesian_score(10),
    )


@pytest.mark.parametrize(
    "top_k,shared_count,overlap_percentage",
    [(1, 0, 0.0), (2, 1, 50.0), (4, 4, 100.0)],
)
def test_top_k_overlap(
    sample_evaluator: RankingEvaluator,
    top_k: int,
    shared_count: int,
    overlap_percentage: float,
) -> None:
    assert sample_evaluator.compare_top_k(top_k) == {
        "top_k": top_k,
        "shared_recipe_count": shared_count,
        "overlap_percentage": overlap_percentage,
    }


def test_rank_improvements(sample_evaluator: RankingEvaluator) -> None:
    rank_changes = sample_evaluator.compare_rank_changes()
    improvements = rank_changes.set_index("recipe_id")["rank_improvement"].to_dict()

    assert improvements == {101: -2, 102: 1, 103: 1, 104: 0}


def test_top_k_rating_counts(sample_evaluator: RankingEvaluator) -> None:
    comparison = sample_evaluator.compare_top_k_rating_counts(2).set_index(
        "ranking_method"
    )

    assert comparison.loc["naive", "median_rating_count"] == 50.5
    assert comparison.loc["bayesian", "median_rating_count"] == 60.0
    assert comparison.loc["naive", "min_rating_count"] == 1
    assert comparison.loc["bayesian", "min_rating_count"] == 20


@pytest.mark.parametrize("top_k", [0, -1, 5, 1.5, True])
def test_invalid_top_k(
    sample_evaluator: RankingEvaluator,
    top_k: float | bool,
) -> None:
    with pytest.raises(ValueError, match="top_k"):
        sample_evaluator.compare_top_k(top_k)


@pytest.mark.parametrize(
    "column,replacement,error_message",
    [
        ("recipe_id", 999, "same recipe IDs"),
        ("rank", 2, "consecutive"),
        ("valid_rating_count", 999, "rating counts"),
    ],
)
def test_inconsistent_rankings_are_rejected(
    sample_evaluator: RankingEvaluator,
    column: str,
    replacement: int,
    error_message: str,
) -> None:
    invalid_ranking = sample_evaluator.bayesian_ranking.copy()
    invalid_ranking.loc[0, column] = replacement

    with pytest.raises(ValueError, match=error_message):
        RankingEvaluator(sample_evaluator.naive_ranking, invalid_ranking)


def test_missing_columns_are_rejected(
    sample_evaluator: RankingEvaluator,
) -> None:
    invalid_ranking = sample_evaluator.bayesian_ranking.drop(columns="ranking_score")

    with pytest.raises(ValueError, match="Missing ranking columns"):
        RankingEvaluator(sample_evaluator.naive_ranking, invalid_ranking)


def test_smoothing_sensitivity(
    sample_ranker: RecipeRanker,
    sample_evaluator: RankingEvaluator,
) -> None:
    results = sample_evaluator.evaluate_smoothing_sensitivity(
        recipe_ranker=sample_ranker,
        smoothing_strengths=[0.1, 1, 10],
        top_k=1,
    )

    assert results["smoothing_strength"].tolist() == [0.1, 1, 10]
    assert results["overlap_with_naive_percentage"].tolist() == [100.0, 0.0, 0.0]
    assert results["median_top_k_rating_count"].tolist() == [1.0, 100.0, 100.0]


def test_empty_smoothing_strengths_are_rejected(
    sample_ranker: RecipeRanker,
    sample_evaluator: RankingEvaluator,
) -> None:
    with pytest.raises(ValueError, match="at least one"):
        sample_evaluator.evaluate_smoothing_sensitivity(
            recipe_ranker=sample_ranker,
            smoothing_strengths=[],
            top_k=1,
        )


@pytest.mark.parametrize(
    "column,replacement",
    [
        ("recipe_id", 999),
        ("valid_rating_count", 2),
        ("mean_valid_rating", 4.5),
    ],
)
def test_smoothing_sensitivity_rejects_mismatched_ranker(
    sample_ranker: RecipeRanker,
    sample_evaluator: RankingEvaluator,
    column: str,
    replacement: float,
) -> None:
    modified_summary = sample_ranker.recipe_rating_summary.copy()
    modified_summary.loc[0, column] = replacement

    mismatched_ranker = RecipeRanker(modified_summary)

    with pytest.raises(ValueError, match="must match"):
        sample_evaluator.evaluate_smoothing_sensitivity(
            recipe_ranker=mismatched_ranker,
            smoothing_strengths=[10],
            top_k=1,
        )
