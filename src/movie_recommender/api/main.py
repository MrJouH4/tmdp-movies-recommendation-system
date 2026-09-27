from pathlib import Path

from fastapi import FastAPI, HTTPException

from movie_recommender.api.schemas import (
    RecommendationRequest,
    RecommendationResponse,
)
from movie_recommender.services.recommendation import (
    RecommendationService,
)


ROOT_DIR = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = ROOT_DIR / "artifacts"


app = FastAPI(
    title="Movie Recommendation API",
    version="0.1.0",
)


service = RecommendationService(ARTIFACTS_DIR)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/recommend",
    response_model=RecommendationResponse,
)
def recommend(request: RecommendationRequest) -> RecommendationResponse:
    try:
        recommendations = service.recommend(
            movie_title=request.movie_title,
            top_k=request.top_k,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    results = [
        {
            "title": row["title"],
            "similarity": float(row["similarity"]),
        }
        for _, row in recommendations.iterrows()
    ]

    return RecommendationResponse(
        recommendations=results,
    )