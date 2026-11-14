import pandas as pd

from rate_my_recipe.ranking import RecipeRanker


class RankingEvaluator:
    """Compare recipe leaderboards and evaluate ranking sensitivity."""

    def __init__(
        self,
        naive_ranking: pd.DataFrame,
        bayesian_ranking: pd.DataFrame,
    ) -> None:
        """Validate and store the naive and Bayesian-style rankings."""
        required_columns = {
            "recipe_id",
            "rank",
            "ranking_score",
            "valid_rating_count",
        }

        for ranking in (naive_ranking, bayesian_ranking):
            missing_columns = required_columns - set(ranking.columns)

            if missing_columns:
                raise ValueError(f"Missing ranking columns: {sorted(missing_columns)}")

            if ranking["recipe_id"].duplicated().any():
                raise ValueError("Recipe IDs must be unique")

            if not ranking["rank"].tolist() == list(range(1, len(ranking) + 1)):
                raise ValueError("Ranks must be consecutive from 1")

        if set(naive_ranking["recipe_id"]) != set(bayesian_ranking["recipe_id"]):
            raise ValueError("Rankings must contain the same recipe IDs")

        naive_counts = naive_ranking.set_index("recipe_id")[
            "valid_rating_count"
        ].sort_index()

        bayesian_counts = bayesian_ranking.set_index("recipe_id")[
            "valid_rating_count"
        ].sort_index()

        if not naive_counts.equals(bayesian_counts):
            raise ValueError("Rankings must agree on valid rating counts")

        self.naive_ranking = naive_ranking.copy()
        self.bayesian_ranking = bayesian_ranking.copy()

    def compare_top_k(self, top_k: int = 20) -> dict:
        """Return the shared recipe count and percentage for a top-k cutoff."""
        self._validate_top_k(top_k)

        naive_top_ids = set(self.naive_ranking.head(top_k)["recipe_id"])
        bayesian_top_ids = set(self.bayesian_ranking.head(top_k)["recipe_id"])

        shared_recipe_count = len(naive_top_ids.intersection(bayesian_top_ids))

        return {
            "top_k": top_k,
            "shared_recipe_count": shared_recipe_count,
            "overlap_percentage": 100 * shared_recipe_count / top_k,
        }

    def compare_rank_changes(self) -> pd.DataFrame:
        """Return per-recipe rank changes between the two methods."""
        naive_positions = self.naive_ranking[
            ["recipe_id", "rank", "ranking_score"]
        ].rename(
            columns={
                "rank": "naive_rank",
                "ranking_score": "naive_score",
            }
        )

        bayesian_positions = self.bayesian_ranking[
            [
                "recipe_id",
                "rank",
                "ranking_score",
                "valid_rating_count",
            ]
        ].rename(
            columns={
                "rank": "bayesian_rank",
                "ranking_score": "bayesian_score",
            }
        )

        rank_comparison = naive_positions.merge(
            bayesian_positions,
            on="recipe_id",
            validate="one_to_one",
        )

        rank_comparison["rank_improvement"] = (
            rank_comparison["naive_rank"] - rank_comparison["bayesian_rank"]
        )

        return rank_comparison.sort_values(
            "rank_improvement",
            ascending=False,
        ).reset_index(drop=True)

    def compare_top_k_rating_counts(
        self,
        top_k: int = 20,
    ) -> pd.DataFrame:
        """Summarize rating counts among top-ranked recipes."""
        self._validate_top_k(top_k)

        ranking_summaries = []

        for ranking_name, ranking in (
            ("naive", self.naive_ranking),
            ("bayesian", self.bayesian_ranking),
        ):
            top_rating_counts = ranking.head(top_k)["valid_rating_count"]

            ranking_summaries.append(
                {
                    "ranking_method": ranking_name,
                    "mean_rating_count": top_rating_counts.mean(),
                    "median_rating_count": top_rating_counts.median(),
                    "min_rating_count": top_rating_counts.min(),
                    "max_rating_count": top_rating_counts.max(),
                }
            )

        return pd.DataFrame(ranking_summaries)

    def evaluate_smoothing_sensitivity(
        self,
        recipe_ranker: RecipeRanker,
        smoothing_strengths: list[float],
        top_k: int = 20,
    ) -> pd.DataFrame:
        """Evaluate top-k behavior across smoothing strengths."""
        self._validate_top_k(top_k)

        if not smoothing_strengths:
            raise ValueError("Provide at least one smoothing strength")

        expected_naive_ranking = recipe_ranker.rank_by_mean_rating()

        comparison_columns = [
            "recipe_id",
            "valid_rating_count",
            "ranking_score",
        ]

        if not expected_naive_ranking[comparison_columns].equals(
            self.naive_ranking[comparison_columns].reset_index(drop=True)
        ):
            raise ValueError("recipe_ranker must match the evaluator's naive ranking")

        baseline_top_ids = set(self.naive_ranking.head(top_k)["recipe_id"])

        sensitivity_results = []

        for smoothing_strength in smoothing_strengths:
            weighted_ranking = recipe_ranker.rank_by_bayesian_score(
                smoothing_strength=smoothing_strength
            )

            weighted_top_ids = set(weighted_ranking.head(top_k)["recipe_id"])

            shared_recipe_count = len(baseline_top_ids.intersection(weighted_top_ids))

            sensitivity_results.append(
                {
                    "smoothing_strength": smoothing_strength,
                    "overlap_with_naive_percentage": (
                        100 * shared_recipe_count / top_k
                    ),
                    "median_top_k_rating_count": (
                        weighted_ranking.head(top_k)["valid_rating_count"].median()
                    ),
                }
            )

        return pd.DataFrame(sensitivity_results)

    def _validate_top_k(self, top_k: int) -> None:
        """Reject invalid top-k cutoffs."""
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or not 1 <= top_k <= len(self.naive_ranking)
        ):
            raise ValueError(
                "top_k must be an integer between 1 and the number of ranked recipes"
            )
