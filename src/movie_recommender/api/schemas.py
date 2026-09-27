from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    movie_title: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)


class MovieRecommendation(BaseModel):
    title: str
    similarity: float


class RecommendationResponse(BaseModel):
    recommendations: list[MovieRecommendation]