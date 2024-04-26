import util
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET", "POST"])
def recommend():
    movie_name = request.form["movie_name"]
    similar_movies = util.recommend(movie_name)
    print(similar_movies)
    return jsonify(similar_movies)


@app.route("/suggestions", methods=["GET", "POST"])
def suggestions():
    query = request.args.get('query')
    return util.get_movie_names(query)


if __name__ == "__main__":
    util.loading_artifacts()
    app.run(port=5000)
