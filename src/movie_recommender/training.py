from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import csr_matrix

from movie_recommender.features.text_features import MovieFeatureBuilder


def fit_and_save(
    movies: pd.DataFrame,
    output_dir: Path,
    max_features: int = 10_000,
    ngram_range: tuple[int, int] = (1, 2),
) -> csr_matrix:
    """Fit TF-IDF features and write the three serving artifacts."""
    feature_builder = MovieFeatureBuilder(
        max_features=max_features,
        ngram_range=ngram_range,
    )
    tfidf_matrix = feature_builder.fit_transform(movies)

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(feature_builder.vectorizer, output_dir / "vectorizer.joblib")
    joblib.dump(tfidf_matrix, output_dir / "tfidf_matrix.joblib")
    joblib.dump(movies, output_dir / "movies.joblib")

    return tfidf_matrix