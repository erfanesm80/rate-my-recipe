import pandas as pd


class RecipeDataPreparer:
    def __init__(
        self,
        recipes: pd.DataFrame,
        interactions: pd.DataFrame,
    ) -> None:
        self.recipes = recipes
        self.interactions = interactions

    def get_valid_rating_interactions(self) -> pd.DataFrame:
        return self.interactions.loc[
            self.interactions["rating"].isin(list(range(1, 6)))
        ].copy()

    def validate_input_data(self) -> None:
        required_recipe_columns = {
            "id",
            "name",
            "minutes",
            "n_ingredients",
        }
        required_interaction_columns = {
            "user_id",
            "recipe_id",
            "rating",
        }

        missing_recipe_columns = required_recipe_columns - set(self.recipes.columns)
        missing_interaction_columns = required_interaction_columns - set(
            self.interactions.columns
        )

        if missing_recipe_columns:
            raise ValueError(
                f"Missing recipe columns: {sorted(missing_recipe_columns)}"
            )

        if missing_interaction_columns:
            raise ValueError(
                f"Missing interaction columns: {sorted(missing_interaction_columns)}"
            )

        if self.recipes["id"].isna().any():
            raise ValueError("Recipe IDs must not contain missing values")

        if self.recipes["id"].duplicated().any():
            raise ValueError("Recipe IDs must be unique")

        if self.interactions["recipe_id"].isna().any():
            raise ValueError("Interaction recipe IDs must not be missing")

        unknown_recipe_ids = ~self.interactions["recipe_id"].isin(self.recipes["id"])

        if unknown_recipe_ids.any():
            raise ValueError("Interactions reference unknown recipe IDs")

        invalid_ratings = self.interactions["rating"].isna() | ~self.interactions[
            "rating"
        ].isin(range(6))

        if invalid_ratings.any():
            raise ValueError("Ratings must be integers from 0 to 5")

    def build_recipe_rating_summary(self) -> pd.DataFrame:
        self.validate_input_data()
        valid_rating_interactions = self.get_valid_rating_interactions()

        valid_rating_statistics = (
            valid_rating_interactions.groupby("recipe_id")["rating"]
            .agg(
                valid_rating_count="size",
                mean_valid_rating="mean",
            )
            .reset_index()
        )

        interaction_counts = (
            self.interactions.groupby("recipe_id")
            .size()
            .rename("interaction_count")
            .reset_index()
        )

        recipe_metadata = self.recipes[
            ["id", "name", "minutes", "n_ingredients"]
        ].rename(
            columns={
                "id": "recipe_id",
                "name": "recipe_name",
                "minutes": "preparation_time_minutes",
                "n_ingredients": "ingredient_count",
            }
        )

        recipe_rating_summary = recipe_metadata.merge(
            interaction_counts,
            on="recipe_id",
            how="left",
            validate="one_to_one",
        ).merge(
            valid_rating_statistics,
            on="recipe_id",
            how="left",
            validate="one_to_one",
        )

        count_columns = [
            "interaction_count",
            "valid_rating_count",
        ]

        recipe_rating_summary[count_columns] = (
            recipe_rating_summary[count_columns].fillna(0).astype(int)
        )

        return recipe_rating_summary
