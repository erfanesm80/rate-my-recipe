import numpy as np
import pandas as pd


class RecipeRanker:
    def __init__(self, recipe_rating_summary: pd.DataFrame) -> None:
        required_columns = {
            "recipe_id",
            "valid_rating_count",
            "mean_valid_rating",
        }

        missing_columns = required_columns - set(recipe_rating_summary.columns)

        if missing_columns:
            raise ValueError(f"Missing ranking columns: {sorted(missing_columns)}")

        if recipe_rating_summary["recipe_id"].isna().any():
            raise ValueError("Recipe IDs must not be missing")

        if recipe_rating_summary["recipe_id"].duplicated().any():
            raise ValueError("Recipe IDs must be unique")

        recipe_rating_counts = recipe_rating_summary["valid_rating_count"]
        recipe_mean_ratings = recipe_rating_summary["mean_valid_rating"]

        if (
            recipe_rating_counts.isna().any()
            or recipe_rating_counts.lt(0).any()
            or not recipe_rating_counts.mod(1).eq(0).all()
        ):
            raise ValueError("Valid rating counts must be nonnegative integers")

        valid_mean_ratings = recipe_mean_ratings.between(1, 5)

        invalid_mean_ratings = (recipe_rating_counts.gt(0) & ~valid_mean_ratings) | (
            recipe_rating_counts.eq(0) & recipe_mean_ratings.notna()
        )

        if invalid_mean_ratings.any():
            raise ValueError(
                "Mean ratings must be between 1 and 5 for rated recipes "
                "and missing for unrated recipes"
            )

        self.recipe_rating_summary = recipe_rating_summary.copy()

    def calculate_global_mean_rating(self) -> float:
        rated_recipes = self.recipe_rating_summary.loc[
            self.recipe_rating_summary["valid_rating_count"].gt(0)
        ]

        total_valid_ratings = rated_recipes["valid_rating_count"].sum()

        if total_valid_ratings == 0:
            raise ValueError("Cannot rank recipes without valid ratings")

        weighted_rating_sum = (
            rated_recipes["valid_rating_count"] * rated_recipes["mean_valid_rating"]
        ).sum()

        return float(weighted_rating_sum / total_valid_ratings)

    def rank_by_mean_rating(self) -> pd.DataFrame:
        rated_recipes = self.recipe_rating_summary.loc[
            self.recipe_rating_summary["valid_rating_count"].gt(0)
        ].copy()

        rated_recipes["ranking_score"] = rated_recipes["mean_valid_rating"]

        return self._sort_ranked_recipes(rated_recipes)

    def rank_by_bayesian_score(
        self,
        smoothing_strength: float = 10.0,
    ) -> pd.DataFrame:
        if not np.isfinite(smoothing_strength) or smoothing_strength <= 0:
            raise ValueError("Smoothing strength must be a positive finite number")

        global_mean_rating = self.calculate_global_mean_rating()

        rated_recipes = self.recipe_rating_summary.loc[
            self.recipe_rating_summary["valid_rating_count"].gt(0)
        ].copy()

        rating_counts = rated_recipes["valid_rating_count"]

        rated_recipes["ranking_score"] = (
            rating_counts * rated_recipes["mean_valid_rating"]
            + smoothing_strength * global_mean_rating
        ) / (rating_counts + smoothing_strength)

        return self._sort_ranked_recipes(rated_recipes)

    def _sort_ranked_recipes(
        self,
        rated_recipes: pd.DataFrame,
    ) -> pd.DataFrame:
        ranked_recipes = rated_recipes.sort_values(
            by=[
                "ranking_score",
                "valid_rating_count",
                "recipe_id",
            ],
            ascending=[False, False, True],
            kind="stable",
        ).reset_index(drop=True)

        ranked_recipes.insert(
            0,
            "rank",
            np.arange(1, len(ranked_recipes) + 1),
        )

        return ranked_recipes
