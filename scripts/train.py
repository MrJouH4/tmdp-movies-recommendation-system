import argparse
import logging
import subprocess
import time
from pathlib import Path

import mlflow
import yaml
from mlflow import MlflowClient

from movie_recommender.config import load_config
from movie_recommender.data.loader import MovieDataLoader
from movie_recommender.evaluation import decide_promotion, evaluate_recommender
from movie_recommender.registry import get_champion_metric, register_and_promote
from movie_recommender.training import fit_and_save

ROOT_DIR = Path(__file__).resolve().parents[1]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def read_data_version(data_path: Path) -> str:
    """Return the DVC md5 of the dataset, so a run points at exact data."""
    dvc_file = data_path.with_name(data_path.name + ".dvc")
    if not dvc_file.exists():
        return "untracked"
    meta = yaml.safe_load(dvc_file.read_text(encoding="utf-8"))
    return meta["outs"][0]["md5"]


def read_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT_DIR,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the recommender.")
    parser.add_argument("--config", type=Path, default=ROOT_DIR / "configs/config.yaml")
    parser.add_argument(
        "--register",
        action="store_true",
        help="Gate the model, compare with the champion, register if better.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    data_path = ROOT_DIR / cfg.raw_data_path
    data_version = read_data_version(data_path)
    # In register mode the serving artifacts/ folder must not be touched by a
    # model that may be rejected. It is only updated by scripts/pull_model.py.
    if args.register:
        output_dir = ROOT_DIR / "build" / "candidate"
    else:
        output_dir = ROOT_DIR / cfg.artifacts_dir

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run() as run:
        logger.info("MLflow run: %s", run.info.run_id)

        movies = MovieDataLoader(data_path).load()
        logger.info("Loaded %d movies (data version %s).", len(movies), data_version)

        mlflow.log_params(
            {
                "max_features": cfg.features.max_features,
                "ngram_range": str(cfg.features.ngram_range),
                "stop_words": "english",
                "eval_top_k": cfg.evaluation.top_k,
                "eval_sample_size": cfg.evaluation.sample_size,
                "eval_random_state": cfg.evaluation.random_state,
                "data_path": cfg.raw_data_path,
                "data_version": data_version,
            }
        )
        mlflow.set_tags({"git_commit": read_git_commit()})

        started = time.perf_counter()
        tfidf_matrix = fit_and_save(
            movies,
            output_dir,
            max_features=cfg.features.max_features,
            ngram_range=cfg.features.ngram_range,
        )
        train_seconds = time.perf_counter() - started
        logger.info("TF-IDF matrix shape: %s", tfidf_matrix.shape)

        metrics = evaluate_recommender(
            movies,
            tfidf_matrix,
            top_k=cfg.evaluation.top_k,
            sample_size=cfg.evaluation.sample_size,
            random_state=cfg.evaluation.random_state,
        )
        metrics.update(
            {
                "n_movies": float(tfidf_matrix.shape[0]),
                "n_features": float(tfidf_matrix.shape[1]),
                "train_seconds": train_seconds,
            }
        )
        mlflow.log_metrics(metrics)
        logger.info("Metrics: %s", metrics)

        mlflow.log_artifacts(str(output_dir), artifact_path="model")
        mlflow.log_artifact(str(args.config), artifact_path="config")

        if not args.register:
            logger.info("Done. Serving artifacts written to %s", output_dir)
            return

        client = MlflowClient()
        metric_name = cfg.promotion.metric
        champion_metric = get_champion_metric(
            client, cfg.mlflow.registered_model_name, metric_name
        )
        decision = decide_promotion(
            candidate=metrics[metric_name],
            champion=champion_metric,
            min_value=cfg.promotion.min_value,
            tolerance=cfg.promotion.tolerance,
        )
        mlflow.set_tags(
            {
                "promotion": "promoted" if decision.promote else "rejected",
                "promotion_reason": decision.reason,
            }
        )
        logger.info("Promotion decision: %s (%s)", decision.promote, decision.reason)

        if decision.promote:
            version = register_and_promote(
                client=client,
                model_name=cfg.mlflow.registered_model_name,
                run_id=run.info.run_id,
                artifact_uri=run.info.artifact_uri,
                metrics={metric_name: metrics[metric_name]},
                data_version=data_version,
            )
            logger.info("Registered version %s as champion.", version)


if __name__ == "__main__":
    main()