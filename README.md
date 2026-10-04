# Movie Recommendation System

A movie recommendation system built with Python using TF-IDF and cosine similarity.

## Project Structure

```text
tmdp-movies-recommendation-system/
├── src/
│   └── movie_recommender/
│       ├── data/
│       ├── features/
│       ├── models/
│       ├── services/
│       └── api/
├── scripts/
│   └── train.py
├── tests/
├── data/
│   └── raw/
├── artifacts/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

## How It Works

1. Load and validate the movie dataset.
2. Combine movie genres and overviews into text features.
3. Build TF-IDF features.
4. Calculate cosine similarity between movies.
5. Save the trained artifacts.
6. Serve recommendations through a FastAPI API.

## Training

Run the training pipeline:

```bash
python scripts/train.py
```

This creates the following artifacts:

```text
artifacts/
├── movies.joblib
├── tfidf_matrix.joblib
└── vectorizer.joblib
```

## Run the API Locally

```bash
uvicorn movie_recommender.api.main:app --reload
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
GET /health
```

Recommendation endpoint:

```text
POST /recommend
```

Example request:

```json
{
    "movie_title": "Inception",
    "top_k": 5
}
```

## Run with Docker

Make sure the training artifacts exist first.

Build and start the container:

```bash
docker compose up --build
```

The API will be available at:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

## Testing

Run all tests:

```bash
pytest
```

Current test suite:

```text
12 passed
```
