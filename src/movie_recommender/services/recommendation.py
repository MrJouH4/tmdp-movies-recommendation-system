from pathlib import Path

import joblib

from movie_recommender.models.recommender import MovieRecommender


class RecommendationService:
    """Serve movie recommendations using trained artifacts."""

    def __init__(self, artifacts_dir: Path) -> None:
        self.artifacts_dir = artifacts_dir

        self.movies = joblib.load(
            self.artifacts_dir / "movies.joblib"
        )

        self.tfidf_matrix = joblib.load(
            self.artifacts_dir / "tfidf_matrix.joblib"
        )

        self.vectorizer = joblib.load(
            self.artifacts_dir / "vectorizer.joblib"
        )

        self.recommender = MovieRecommender(
            movies=self.movies,
            tfidf_matrix=self.tfidf_matrix,
        )

    def recommend(
        self,
        movie_title: str,
        top_k: int = 10,
    ):
        return self.recommender.recommend(
            movie_title=movie_title,
            top_k=top_k,
        )