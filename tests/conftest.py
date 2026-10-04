import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from movie_recommender.training import fit_and_save


def _tiny_catalog() -> pd.DataFrame:
    genres = ["Action,Sci-Fi", "Romance,Drama", "Comedy", "Horror,Thriller"]
    titles = ["Inception"] + [f"Movie {i}" for i in range(1, 20)]
    return pd.DataFrame(
        {
            "id": range(len(titles)),
            "title": titles,
            "genre": [genres[i % 4] for i in range(len(titles))],
            "overview": [
                f"{genres[i % 4]} story number {i} about a hero and a quest"
                for i in range(len(titles))
            ],
        }
    )


# Tests must not depend on a locally trained artifacts/ folder, because CI
# starts from a clean checkout. Build a tiny model once and point the API at
# it. This runs at import time, before test modules import the FastAPI app,
# which loads its artifacts at import.
_ARTIFACTS = Path(tempfile.mkdtemp(prefix="recsys-artifacts-"))
fit_and_save(_tiny_catalog(), _ARTIFACTS)
os.environ["ARTIFACTS_DIR"] = str(_ARTIFACTS)


@pytest.fixture(scope="session")
def artifacts_dir() -> Path:
    return _ARTIFACTS