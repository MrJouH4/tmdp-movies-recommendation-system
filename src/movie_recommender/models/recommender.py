import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    """Generate movie recommendations based on TF-IDF similarity."""

    def __init__(
        self,
        movies: pd.DataFrame,
        tfidf_matrix: csr_matrix,
    ) -> None:
        self.movies = movies.reset_index(drop=True)
        self.tfidf_matrix = tfidf_matrix

    def _find_movie_index(self, movie_title: str) -> int:
        matches = self.movies[
            self.movies["title"].str.lower() == movie_title.lower()
        ]

        if matches.empty:
            raise ValueError(f"Movie not found: {movie_title}")

        return int(matches.index[0])

    def recommend(
        self,
        movie_title: str,
        top_k: int = 10,
    ) -> pd.DataFrame:
        movie_index = self._find_movie_index(movie_title)

        query_vector = self.tfidf_matrix[movie_index]

        similarities = cosine_similarity(
            query_vector,
            self.tfidf_matrix,
        ).flatten()

        similarities[movie_index] = -1

        top_indices = similarities.argsort()[-top_k:][::-1]

        recommendations = self.movies.iloc[top_indices].copy()
        recommendations["similarity"] = similarities[top_indices]

        return recommendations