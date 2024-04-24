import pandas as pd
import pickle
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

__model = None
__data = None
__tfidf_matrix = None


def loading_artifacts():
    print("Loading Artifacts.. Start")

    global __data
    __data = pd.read_csv("../model/data/top10K-TMDB-movies_with_tags.csv")

    global __model, __tfidf_matrix
    __model = TfidfVectorizer(stop_words="english")
    __tfidf_matrix = __model.fit_transform(__data['tags'])

    print("Loading Artifacts.. Done")


def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=642e8c4c1bc08c0f20e9e0362c801704&language=en-US"
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path


def recommend(movie_title):
    movie_matrix = __model.transform(__data[__data.title == movie_title].tags)
    similarity_scores = cosine_similarity(movie_matrix, __tfidf_matrix)
    top_similar_indices = similarity_scores.argsort()[0][::-1][:10]
    similar_movies_info = []

    for idx in top_similar_indices:
        movie_name = __data.iloc[idx]["title"]
        movie_id = int(__data.iloc[idx]["id"])
        poster_path = fetch_poster(movie_id)
        similar_movies_info.append({
            "id": movie_id,
            "movie_name": movie_name,
            "poster_path": poster_path
        })

    return similar_movies_info


if __name__ == "__main__":
    loading_artifacts()
