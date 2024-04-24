import util
from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def recommend():
    movie_name = request.form["movie_name"]
    similar_movies = util.recommend(movie_name)
    print(similar_movies)
    # return similar_movies
    return jsonify(similar_movies)


if __name__ == "__main__":
    util.loading_artifacts()
    app.run(port=5000)
