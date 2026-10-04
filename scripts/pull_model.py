import logging
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from movie_recommender.config import load_config
from movie_recommender.registry import download_champion

ROOT_DIR = Path(__file__).resolve().parents[1]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def main() -> None:
    cfg = load_config(ROOT_DIR / "configs" / "config.yaml")

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    client = MlflowClient()

    version = download_champion(
        client,
        cfg.mlflow.registered_model_name,
        ROOT_DIR / cfg.artifacts_dir,
    )
    logger.info("Pulled champion version %s into %s", version, cfg.artifacts_dir)


if __name__ == "__main__":
    main()