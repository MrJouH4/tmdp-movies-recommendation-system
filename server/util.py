import pandas as pd
import pickle
import requests

__model = None
__data = None

def loading_artifacts():
    print("Loading Artifacts.. Start")

    global __data
    __data = pd.read_csv("../model/data/top10K-TMDB-movies.csv")

    global __model
    __model = pickle.load("../model/tfidf.pkl")

    # Save & load npz (tfidf matrix)

    print("Loading Artifacts.. Done")


def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=642e8c4c1bc08c0f20e9e0362c801704&language=en-US"
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path


def recommend(movie_title):
    movie_matrix = __model.transform(data[data.title == movie_title].tags)
    similarity_scores = cosine_similarity(movie_matrix, tfidf_matrix)
    top_similar_indices = similarity_scores.argsort()[0][::-1][:10]
    similar_movies = df.iloc[top_similar_indices]["title"].tolist()
    return similar_movies


if __name__ == "__main__":
    pass