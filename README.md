# Movie Recommendation System

[![CI](https://github.com/MrJouH4/tmdp-movies-recommendation-system/actions/workflows/ci.yml/badge.svg)](https://github.com/MrJouH4/tmdp-movies-recommendation-system/actions/workflows/ci.yml)

A content-based movie recommender (TF-IDF + cosine similarity) served with FastAPI,
built as an end-to-end MLOps learning project: versioned data, tracked experiments,
a gated model registry, CI, and a Dockerized API.

## Architecture

```text
data/raw/movies.csv ──► DVC (pointer file movies.csv.dvc is committed to Git)
        │
        ▼
scripts/train.py ──► MLflow run: params, metrics, artifacts, data_version, git_commit
        │
        ├─ evaluate (offline metrics)
        ├─ quality gate + comparison with the current champion
        ▼
MLflow Model Registry: "movie-recommender" v1, v2, ...  (alias: champion)
        │   scripts/pull_model.py
        ▼
artifacts/ ──► Docker / FastAPI

GitHub Actions on push / PR: install → ruff → pytest → docker build
```

Training and serving are separate: the API never trains. It only loads the artifacts
found in `ARTIFACTS_DIR`.

## Project structure

```text
├── .github/workflows/ci.yml
├── configs/config.yaml          # features, evaluation, MLflow and promotion settings
├── data/raw/
│   ├── movies.csv               # tracked by DVC, not by Git
│   └── movies.csv.dvc           # dataset pointer (md5), tracked by Git
├── scripts/
│   ├── train.py                 # train, evaluate, log to MLflow, optionally register
│   └── pull_model.py            # copy the champion model into artifacts/
├── src/movie_recommender/
│   ├── config.py                # typed config loaded from YAML
│   ├── training.py              # fit TF-IDF and save the three artifacts
│   ├── evaluation.py            # offline metrics and the promotion decision
│   ├── registry.py              # MLflow registry helpers
│   ├── data/                    # loading and validation
│   ├── features/                # TF-IDF feature builder
│   ├── models/                  # recommender (cosine similarity, top-K)
│   ├── services/                # recommendation service
│   └── api/                     # FastAPI app and schemas
├── tests/
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; use source .venv/bin/activate on Linux/macOS
pip install -e ".[dev,api,mlops]"
```

Get the dataset (needs access to the DVC remote configured in `.dvc/config`):

```bash
dvc pull
```

## Training

Development run. Writes serving artifacts directly to `artifacts/`:

```bash
python scripts/train.py
```

Registry run. Trains into `build/candidate/`, evaluates, compares with the current
champion, and registers a new model version only if it is accepted:

```bash
python scripts/train.py --register
python scripts/pull_model.py      # copy the champion into artifacts/
```

`artifacts/` is never overwritten by a rejected model in register mode.

### What each run logs to MLflow

| Kind | Items |
|---|---|
| Parameters | `max_features`, `ngram_range`, evaluation settings, `data_version` (DVC md5) |
| Metrics | `genre_precision_at_k`, `mean_similarity`, `catalog_coverage`, `n_movies`, `n_features`, `train_seconds` |
| Artifacts | the three `.joblib` files and the config used |
| Tags | `git_commit`, `promotion`, `promotion_reason` |

Browse runs and registered versions:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open http://127.0.0.1:5000.

## Evaluation and promotion

There are no user ratings or clicks in this dataset, so there is no ground truth for
"relevant". The offline metric is `genre_precision_at_k`: the share of the top-K
recommendations that share at least one genre with the query movie, averaged over a
fixed random sample of 500 movies.

A candidate is promoted to `champion` when:

1. its score is at least the absolute gate (`promotion.min_value`, currently 0.70), and
2. it is at least as good as the current champion (within `promotion.tolerance`).

Limitations: genre is also an input feature, so this metric is a regression check
between model versions, not a measure of real recommendation quality. When the dataset
changes, the champion's score was computed on the older data, so scores are not
strictly comparable across dataset versions.

## Data versioning (DVC)

`data/raw/movies.csv` is tracked with DVC. Git stores only the small pointer file
`movies.csv.dvc`. Every MLflow run records the dataset md5 as `data_version`.

```bash
dvc status                                   # has the dataset changed?
dvc add data/raw/movies.csv                  # after changing it
git add data/raw/movies.csv.dvc
git commit -m "Update dataset"
dvc push

# restore an older dataset version
git checkout <commit> -- data/raw/movies.csv.dvc
dvc checkout
```

The configured remote is a local folder (`../dvc-storage`).

## Run the API

Locally:

```bash
uvicorn movie_recommender.api.main:app --reload
```

With Docker (needs `artifacts/` to exist, see `pull_model.py`):

```bash
docker compose up --build
```

The API is at http://localhost:8000 (Swagger UI at `/docs`). The container mounts
`./artifacts` read-only and reads the path from `ARTIFACTS_DIR`.

Endpoints:

```text
GET  /health
POST /recommend
```

Example request:

```json
{
    "movie_title": "Inception",
    "top_k": 5
}
```

## Testing and CI

```bash
ruff check src scripts tests
pytest
```

Tests do not need the real dataset or trained artifacts: `tests/conftest.py` builds
a tiny model in a temporary directory. GitHub Actions runs lint and tests on every
push and pull request, then builds the Docker image.

## Current limitations

- Retraining is triggered manually (`python scripts/train.py --register`). It is not
  automated yet.
- The DVC remote and the MLflow backend (`sqlite:///mlflow.db`) are local, so
  training cannot run in CI yet.
- The model is TF-IDF over genre and overview text; there is no personalization.
- The CI Docker job builds the image but does not run it.
