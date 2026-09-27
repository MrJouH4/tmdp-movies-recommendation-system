from pathlib import Path
import joblib
import pandas as pd
from movie_recommender.models.recommender import MovieRecommender


def test_training_artifacts_exist() -> None:
    artifacts_dir = Path("artifacts")

    assert (artifacts_dir / "movies.joblib").exists()
    assert (artifacts_dir / "tfidf_matrix.joblib").exists()
    assert (artifacts_dir / "vectorizer.joblib").exists()


def test_saved_artifacts_can_be_used_for_recommendations() -> None:
    artifacts_dir = Path("artifacts")

    movies = joblib.load(artifacts_dir / "movies.joblib")
    tfidf_matrix = joblib.load(artifacts_dir / "tfidf_matrix.joblib")

    recommender = MovieRecommender(
        movies=movies,
        tfidf_matrix=tfidf_matrix,
    )

    movie_title = movies.iloc[0]["title"]

    recommendations = recommender.recommend(
        movie_title=movie_title,
        top_k=5,
    )

    assert isinstance(recommendations, pd.DataFrame)
    assert len(recommendations) == 5
    assert "similarity" in recommendations.columns
    assert movie_title not in recommendations["title"].values