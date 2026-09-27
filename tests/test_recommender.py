import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from movie_recommender.models.recommender import MovieRecommender


def test_recommender_returns_similar_movies() -> None:
    movies = pd.DataFrame(
        {
            "title": [
                "Space War",
                "Galaxy Battle",
                "Romantic Dinner",
                "Space Journey",
            ],
            "genre": [
                "Action Sci-Fi",
                "Action Sci-Fi",
                "Romance",
                "Adventure Sci-Fi",
            ],
            "overview": [
                "A war in space.",
                "A battle in the galaxy.",
                "A couple enjoys a romantic dinner.",
                "A journey through space.",
            ],
        }
    )

    tags = (
        movies["genre"] + " " + movies["overview"]
    )

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(tags)

    recommender = MovieRecommender(
        movies=movies,
        tfidf_matrix=tfidf_matrix,
    )

    recommendations = recommender.recommend(
        movie_title="Space War",
        top_k=2,
    )

    assert len(recommendations) == 2
    assert "Space War" not in recommendations["title"].values
    assert "similarity" in recommendations.columns