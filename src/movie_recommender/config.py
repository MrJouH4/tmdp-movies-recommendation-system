import os
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class FeatureConfig:
    max_features: int
    ngram_range: tuple[int, int]


@dataclass(frozen=True)
class EvaluationConfig:
    top_k: int
    sample_size: int
    random_state: int


@dataclass(frozen=True)
class MlflowConfig:
    tracking_uri: str
    experiment_name: str
    registered_model_name: str


@dataclass(frozen=True)
class PromotionConfig:
    metric: str
    min_value: float
    tolerance: float


@dataclass(frozen=True)
class Config:
    raw_data_path: str
    artifacts_dir: str
    features: FeatureConfig
    evaluation: EvaluationConfig
    mlflow: MlflowConfig
    promotion: PromotionConfig


def load_config(path: Path) -> Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    features = raw["features"]
    mlflow_cfg = raw["mlflow"]

    return Config(
        raw_data_path=raw["data"]["raw_path"],
        artifacts_dir=raw["artifacts_dir"],
        features=FeatureConfig(
            max_features=features["max_features"],
            ngram_range=tuple(features["ngram_range"]),
        ),
        evaluation=EvaluationConfig(**raw["evaluation"]),
        mlflow=MlflowConfig(
            # Lets CI or another machine point at a different MLflow backend.
            tracking_uri=os.getenv(
                "MLFLOW_TRACKING_URI", mlflow_cfg["tracking_uri"]
            ),
            experiment_name=mlflow_cfg["experiment_name"],
            registered_model_name=mlflow_cfg["registered_model_name"],
        ),
        promotion=PromotionConfig(**raw["promotion"]),
    )