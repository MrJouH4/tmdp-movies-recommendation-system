from pathlib import Path

import pandas as pd

from movie_recommender.services.recommendation import (
    RecommendationService,
)


def test_recommendation_service_returns_recommendations(artifacts_dir: Path) -> None:
    service = RecommendationService(artifacts_dir)

    movie_title = service.movies.iloc[0]["title"]

    recommendations = service.recommend(
        movie_title=movie_title,
        top_k=5,
    )

    assert isinstance(recommendations, pd.DataFrame)
    assert len(recommendations) == 5
    assert "similarity" in recommendations.columns
    assert movie_title not in recommendations["title"].values