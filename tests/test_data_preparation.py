import pandas as pd
import pytest

from rate_my_recipe.data_preparation import RecipeDataPreparer


@pytest.fixture
def sample_preparer() -> RecipeDataPreparer:
    recipes = pd.DataFrame(
        {
            "id": [101, 102, 103],
            "name": ["Recipe A", "Recipe B", "Recipe C"],
            "minutes": [20, 40, 60],
            "n_ingredients": [3, 5, 7],
        }
    )

    interactions = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4, 5],
            "recipe_id": [101, 101, 102, 102, 103],
            "rating": [5, 3, 0, 4, 0],
        }
    )

    return RecipeDataPreparer(recipes, interactions)


def test_valid_rating_filter_excludes_zero(
    sample_preparer: RecipeDataPreparer,
) -> None:
    valid_ratings = sample_preparer.get_valid_rating_interactions()

    assert valid_ratings["rating"].tolist() == [5, 3, 4]


def test_recipe_summary_preserves_unrated_recipes(
    sample_preparer: RecipeDataPreparer,
) -> None:
    summary = sample_preparer.build_recipe_rating_summary()
    summary = summary.set_index("recipe_id")

    assert len(summary) == 3
    assert summary["interaction_count"].to_dict() == {101: 2, 102: 2, 103: 1}
    assert summary["valid_rating_count"].to_dict() == {101: 2, 102: 1, 103: 0}
    assert summary.loc[[101, 102], "mean_valid_rating"].tolist() == [4.0, 4.0]
    assert pd.isna(summary.loc[103, "mean_valid_rating"])


def test_duplicate_recipe_ids_are_rejected(
    sample_preparer: RecipeDataPreparer,
) -> None:
    sample_preparer.recipes = pd.concat(
        [
            sample_preparer.recipes,
            sample_preparer.recipes.iloc[[0]],
        ]
    )

    with pytest.raises(ValueError, match="unique"):
        sample_preparer.build_recipe_rating_summary()


def test_unknown_recipe_ids_are_rejected(
    sample_preparer: RecipeDataPreparer,
) -> None:
    sample_preparer.interactions.loc[0, "recipe_id"] = 999

    with pytest.raises(ValueError, match="unknown"):
        sample_preparer.build_recipe_rating_summary()


@pytest.mark.parametrize("invalid_rating", [-1, 6, 2.5, None])
def test_invalid_ratings_are_rejected(
    sample_preparer: RecipeDataPreparer,
    invalid_rating: float | None,
) -> None:
    sample_preparer.interactions["rating"] = sample_preparer.interactions[
        "rating"
    ].astype(float)
    sample_preparer.interactions.loc[0, "rating"] = invalid_rating

    with pytest.raises(ValueError, match="Ratings"):
        sample_preparer.build_recipe_rating_summary()


def test_empty_interactions_preserve_all_recipes(
    sample_preparer: RecipeDataPreparer,
) -> None:
    sample_preparer.interactions = sample_preparer.interactions.iloc[0:0]

    summary = sample_preparer.build_recipe_rating_summary()

    assert len(summary) == 3
    assert summary["interaction_count"].eq(0).all()
    assert summary["valid_rating_count"].eq(0).all()
    assert summary["mean_valid_rating"].isna().all()


def test_missing_required_columns_are_rejected(
    sample_preparer: RecipeDataPreparer,
) -> None:
    sample_preparer.recipes = sample_preparer.recipes.drop(columns="n_ingredients")

    with pytest.raises(ValueError, match="Missing recipe columns"):
        sample_preparer.build_recipe_rating_summary()


def test_recipe_tag_index_normalizes_and_deduplicates(
    sample_preparer: RecipeDataPreparer,
) -> None:
    sample_preparer.recipes["tags"] = [
        "[' Vegetarian ', 'quick', 'vegetarian']",
        "['quick']",
        "[]",
    ]

    tag_index = sample_preparer.build_recipe_tag_index()

    assert tag_index.to_dict("records") == [
        {"recipe_id": 101, "tag": "vegetarian"},
        {"recipe_id": 101, "tag": "quick"},
        {"recipe_id": 102, "tag": "quick"},
    ]


def test_missing_tags_column_is_rejected(
    sample_preparer: RecipeDataPreparer,
) -> None:
    with pytest.raises(ValueError, match="Missing recipe columns"):
        sample_preparer.build_recipe_tag_index()


@pytest.mark.parametrize(
    "invalid_tags,expected_exception",
    [
        ("not a list", ValueError),
        ("['vegetarian', 123]", ValueError),
        (None, TypeError),
    ],
)
def test_invalid_tag_data_is_rejected(
    sample_preparer: RecipeDataPreparer,
    invalid_tags: str | None,
    expected_exception: type[Exception],
) -> None:
    sample_preparer.recipes["tags"] = [
        invalid_tags,
        "['quick']",
        "[]",
    ]

    with pytest.raises(expected_exception, match="Recipe tags"):
        sample_preparer.build_recipe_tag_index()
