import pandas as pd
import pytest

from rate_my_recipe.ranking import RecipeRanker


@pytest.fixture
def sample_ranker() -> RecipeRanker:
    recipe_ratings = pd.DataFrame(
        {
            "recipe_id": [101, 102, 103, 104, 105],
            "valid_rating_count": [1, 100, 20, 100, 0],
            "mean_valid_rating": [5.0, 4.8, 4.6, 3.0, float("nan")],
        }
    )

    return RecipeRanker(recipe_ratings)
