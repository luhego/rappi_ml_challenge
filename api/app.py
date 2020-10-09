from flask import Flask, jsonify, request

from predict import run_prediction

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    prediction = run_prediction(data)

    return jsonify({"data": prediction})


if __name__ == "__main__":
    app.run(threaded=True, port=5000)
