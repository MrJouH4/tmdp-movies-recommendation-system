import util
from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route("/hi", methods=["GET", "POST"])
def fetch_poster():
    return "HI"


if __name__ == "__main__":
    fetch_poster()
    app.run(port=5000)
