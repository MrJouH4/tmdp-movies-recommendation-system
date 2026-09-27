import pandas as pd

from movie_recommender.features.text_features import MovieFeatureBuilder


def test_build_tags_combines_genre_and_overview() -> None:
    data = pd.DataFrame(
        {
            "genre": ["Action", "Comedy"],
            "overview": [
                "A group of heroes fight.",
                "A funny story.",
            ],
        }
    )

    builder = MovieFeatureBuilder()

    tags = builder.build_tags(data)

    assert tags.iloc[0] == "Action A group of heroes fight."
    assert tags.iloc[1] == "Comedy A funny story."


def test_fit_transform_returns_tfidf_matrix() -> None:
    data = pd.DataFrame(
        {
            "genre": ["Action", "Comedy"],
            "overview": [
                "A group of heroes fight.",
                "A funny story.",
            ],
        }
    )

    builder = MovieFeatureBuilder()

    matrix = builder.fit_transform(data)

    assert matrix.shape[0] == 2
    assert matrix.shape[1] > 0