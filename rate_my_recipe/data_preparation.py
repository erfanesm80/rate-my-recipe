import ast

import numpy as np
import pandas as pd


class RecipeDataPreparer:
    """Validate Food.com data and prepare recipe-level datasets."""

    def __init__(
        self,
        recipes: pd.DataFrame,
        interactions: pd.DataFrame,
    ) -> None:
        """Store recipe metadata and user interactions.
        Args:
            recipes: Recipe metadata containing IDs, names, and attributes.
            interactions: User-recipe interactions containing IDs and ratings.
        """
        self.recipes = recipes
        self.interactions = interactions

    def get_valid_rating_interactions(self) -> pd.DataFrame:
        """Select interactions with valid star ratings.
        Returns:
            Copy of interaction rows with ratings from 1 to 5,
            excluding zero ratings.
        """
        return self.interactions.loc[
            self.interactions["rating"].isin(list(range(1, 6)))
        ].copy()

    def validate_input_data(self) -> None:
        """Validate recipe identifiers, references, and rating values.
        Raises:
            ValueError: If required columns are missing, recipe IDs are
                invalid or duplicated, interactions reference unknown
                recipes, or ratings contain invalid values.
        """
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

        ratings = self.interactions["rating"]

        boolean_ratings = ratings.map(lambda value: isinstance(value, (bool, np.bool_)))

        invalid_ratings = ratings.isna() | boolean_ratings | ~ratings.isin(range(6))

        if invalid_ratings.any():
            raise ValueError("Ratings must be integers from 0 to 5")

    def build_recipe_rating_summary(self) -> pd.DataFrame:
        """Aggregate recipe metadata and rating statistics.

        Returns:
            DataFrame containing one row per recipe, including metadata,
            total interactions, valid rating counts, and mean valid ratings.
            Unrated recipes retain zero counts and a missing mean rating.

        Raises:
            ValueError: If required columns, recipe IDs, or ratings are invalid.
        """
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

    def build_recipe_tag_index(self) -> pd.DataFrame:
        """Build a normalized index of recipe-tag assignments.
        Returns:
            DataFrame with unique recipe_id and tag pairs, using
            lowercase tags without surrounding whitespace.

        Raises:
            ValueError: If input data or serialized tag lists are invalid.
            TypeError: If a serialized tag-list value is not a string.
        """
        self.validate_input_data()

        if "tags" not in self.recipes.columns:
            raise ValueError("Missing recipe columns: ['tags']")

        recipe_tags = (
            self.recipes[["id", "tags"]].rename(columns={"id": "recipe_id"}).copy()
        )

        recipe_tags["tag"] = recipe_tags["tags"].map(self._parse_recipe_tags)

        return (
            recipe_tags.drop(columns="tags")
            .explode("tag", ignore_index=True)
            .dropna(subset=["tag"])
            .drop_duplicates(ignore_index=True)
        )

    @staticmethod
    def _parse_recipe_tags(raw_tags: str) -> list[str]:
        """Parse and normalize a serialized list of recipe tags.
        Args:
            raw_tags: String representation of a Python list of tags.

        Returns:
            List of lowercase, stripped tags with empty values removed.

        Raises:
            TypeError: If raw_tags is not a string.
            ValueError: If the literal is malformed or does not contain
                a list of strings.
        """
        if not isinstance(raw_tags, str):
            raise TypeError("Recipe tags must be serialized lists of strings")

        try:
            parsed_tags = ast.literal_eval(raw_tags)
        except (ValueError, SyntaxError) as error:
            raise ValueError(
                "Recipe tags must be serialized lists of strings"
            ) from error

        if not isinstance(parsed_tags, list) or any(
            not isinstance(tag, str) for tag in parsed_tags
        ):
            raise ValueError("Recipe tags must be serialized lists of strings")

        return [tag.strip().lower() for tag in parsed_tags if tag.strip()]
