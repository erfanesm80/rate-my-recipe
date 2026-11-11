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

    def build_recipe_rating_summary(self) -> pd.DataFrame:
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
