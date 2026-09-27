from pathlib import Path
import pandas as pd
import pytest

from movie_recommender.data.loader import MovieDataLoader


def test_loader_returns_dataframe(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "id": [1, 2],
            "title": ["Movie A", "Movie B"],
            "genre": ["Action", "Comedy"],
            "overview": ["An action movie", "A comedy movie"],
        }
    )

    csv_path = tmp_path / "movies.csv"
    data.to_csv(csv_path, index=False)

    loader = MovieDataLoader(csv_path)
    result = loader.load()

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2

def test_loader_raises_error_when_required_column_is_missing(
    tmp_path: Path,
) -> None:
    data = pd.DataFrame(
        {
            "id": [1],
            "title": ["Movie A"],
            "genre": ["Action"],
            # overview is intentionally missing
        }
    )

    csv_path = tmp_path / "movies.csv"
    data.to_csv(csv_path, index=False)

    loader = MovieDataLoader(csv_path)

    with pytest.raises(ValueError, match="Missing required columns"):
        loader.load()