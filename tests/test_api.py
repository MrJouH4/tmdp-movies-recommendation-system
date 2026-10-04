from fastapi.testclient import TestClient

from movie_recommender.api.main import app

client = TestClient(app)


def test_health_check() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_recommend_endpoint() -> None:
    response = client.post(
        "/recommend",
        json={
            "movie_title": "Inception",
            "top_k": 5,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "recommendations" in data
    assert len(data["recommendations"]) == 5

    for recommendation in data["recommendations"]:
        assert "title" in recommendation
        assert "similarity" in recommendation


def test_recommend_endpoint_validates_top_k() -> None:
    response = client.post(
        "/recommend",
        json={
            "movie_title": "Inception",
            "top_k": 100,
        },
    )

    assert response.status_code == 422


def test_recommend_endpoint_returns_404_for_unknown_movie() -> None:
    response = client.post(
        "/recommend",
        json={
            "movie_title": "This Movie Definitely Does Not Exist",
            "top_k": 5,
        },
    )

    assert response.status_code == 404