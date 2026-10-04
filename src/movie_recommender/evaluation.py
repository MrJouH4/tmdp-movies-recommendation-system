from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from movie_recommender.models.recommender import MovieRecommender


def _genres(value: object) -> set[str]:
    if not isinstance(value, str):
        return set()
    return {g.strip().lower() for g in value.split(",") if g.strip()}


def evaluate_recommender(
    movies: pd.DataFrame,
    tfidf_matrix: csr_matrix,
    top_k: int = 10,
    sample_size: int = 500,
    random_state: int = 42,
) -> dict[str, float]:
    """Offline sanity metrics for a content-based recommender.

    There are no user interactions in this project, so there is no ground
    truth for "relevant". We use genre agreement as a cheap proxy.
    Genre is also an input feature, so treat this as a regression check
    between model versions, not as an absolute quality score.
    """
    recommender = MovieRecommender(movies, tfidf_matrix)
    catalog = recommender.movies

    # Titles are the lookup key, so sample only titles that are unique.
    title_key = catalog["title"].str.lower()
    candidates = catalog[~title_key.duplicated(keep=False)]

    rng = np.random.default_rng(random_state)
    n_queries = min(sample_size, len(candidates))
    query_positions = rng.choice(len(candidates), size=n_queries, replace=False)

    precisions: list[float] = []
    similarities: list[float] = []
    recommended_ids: set[int] = set()

    for position in query_positions:
        query = candidates.iloc[position]
        recs = recommender.recommend(query["title"], top_k=top_k)

        query_genres = _genres(query["genre"])
        hits = sum(
            bool(query_genres & _genres(genre)) for genre in recs["genre"]
        )

        precisions.append(hits / top_k)
        similarities.extend(recs["similarity"].tolist())
        recommended_ids.update(int(i) for i in recs.index)

    return {
        "genre_precision_at_k": float(np.mean(precisions)),
        "mean_similarity": float(np.mean(similarities)),
        "catalog_coverage": len(recommended_ids) / len(catalog),
    }


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    reason: str


def decide_promotion(
    candidate: float,
    champion: float | None,
    min_value: float,
    tolerance: float = 0.0,
) -> PromotionDecision:
    """Quality gate first, then compare with the current champion."""
    if candidate < min_value:
        return PromotionDecision(
            False, f"candidate {candidate:.4f} is below the gate {min_value:.4f}"
        )

    if champion is None:
        return PromotionDecision(True, "no champion yet and the gate passed")

    if candidate >= champion - tolerance:
        return PromotionDecision(
            True, f"candidate {candidate:.4f} >= champion {champion:.4f}"
        )

    return PromotionDecision(
        False, f"candidate {candidate:.4f} is worse than champion {champion:.4f}"
    )