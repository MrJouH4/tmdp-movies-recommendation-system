import joblib

from movie_recommender.evaluation import (
    decide_promotion,
    evaluate_recommender,
)


def test_evaluate_recommender_returns_expected_metrics(artifacts_dir) -> None:
    movies = joblib.load(artifacts_dir / "movies.joblib")
    matrix = joblib.load(artifacts_dir / "tfidf_matrix.joblib")

    metrics = evaluate_recommender(
        movies, matrix, top_k=3, sample_size=10, random_state=0
    )

    assert set(metrics) == {
        "genre_precision_at_k",
        "mean_similarity",
        "catalog_coverage",
    }
    assert 0.0 <= metrics["genre_precision_at_k"] <= 1.0
    assert 0.0 < metrics["catalog_coverage"] <= 1.0


def test_promotion_rejects_candidate_below_gate() -> None:
    decision = decide_promotion(candidate=0.6, champion=None, min_value=0.7)
    assert decision.promote is False


def test_promotion_accepts_first_model_that_passes_gate() -> None:
    decision = decide_promotion(candidate=0.8, champion=None, min_value=0.7)
    assert decision.promote is True


def test_promotion_rejects_candidate_worse_than_champion() -> None:
    decision = decide_promotion(candidate=0.75, champion=0.80, min_value=0.7)
    assert decision.promote is False


def test_promotion_accepts_candidate_equal_or_better() -> None:
    assert decide_promotion(0.80, 0.80, 0.7).promote is True
    assert decide_promotion(0.85, 0.80, 0.7).promote is True