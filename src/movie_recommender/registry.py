from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

CHAMPION_ALIAS = "champion"


def get_champion_metric(
    client: MlflowClient,
    model_name: str,
    metric: str,
) -> float | None:
    """Return the metric of the run behind the current champion, if any."""
    try:
        champion = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
    except MlflowException:
        return None  # model or alias does not exist yet

    run = client.get_run(champion.run_id)
    return run.data.metrics.get(metric)


def register_and_promote(
    client: MlflowClient,
    model_name: str,
    run_id: str,
    artifact_uri: str,
    metrics: dict[str, float],
    data_version: str,
) -> str:
    """Create a new model version from a run's artifacts and make it champion."""
    try:
        client.get_registered_model(model_name)
    except MlflowException:
        client.create_registered_model(model_name)

    version = client.create_model_version(
        name=model_name,
        source=f"{artifact_uri}/model",
        run_id=run_id,
    )

    for name, value in metrics.items():
        client.set_model_version_tag(model_name, version.version, name, f"{value:.4f}")
    client.set_model_version_tag(
        model_name, version.version, "data_version", data_version
    )

    client.set_registered_model_alias(model_name, CHAMPION_ALIAS, version.version)
    return version.version


def download_champion(
    client: MlflowClient,
    model_name: str,
    destination: Path,
) -> str:
    """Copy the champion's joblib artifacts into `destination`."""
    champion = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)

    local_dir = Path(
        mlflow.artifacts.download_artifacts(artifact_uri=champion.source)
    )

    destination.mkdir(parents=True, exist_ok=True)
    for file in local_dir.glob("*.joblib"):
        (destination / file.name).write_bytes(file.read_bytes())

    return champion.version