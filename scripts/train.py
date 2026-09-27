from pathlib import Path
import logging

import joblib

from movie_recommender.data.loader import MovieDataLoader
from movie_recommender.features.text_features import MovieFeatureBuilder


ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT_DIR / "data" / "raw" / "movies.csv"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

def main() -> None:
    logger.info("Starting training pipeline.")

    logger.info("Loading movie data from %s", DATA_PATH)
    loader = MovieDataLoader(DATA_PATH)
    movies = loader.load()

    logger.info("Loaded %d movies.", len(movies))

    logger.info("Building TF-IDF features.")
    feature_builder = MovieFeatureBuilder()
    tfidf_matrix = feature_builder.fit_transform(movies)

    logger.info(
        "TF-IDF matrix shape: %s",
        tfidf_matrix.shape,
    )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Saving training artifacts to %s", ARTIFACTS_DIR)

    joblib.dump(
        feature_builder.vectorizer,
        ARTIFACTS_DIR / "vectorizer.joblib",
    )

    joblib.dump(
        tfidf_matrix,
        ARTIFACTS_DIR / "tfidf_matrix.joblib",
    )

    joblib.dump(
        movies,
        ARTIFACTS_DIR / "movies.joblib",
    )
    
    logger.info("Training pipeline completed successfully.")


if __name__ == "__main__":
    main()