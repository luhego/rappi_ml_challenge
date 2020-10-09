from flask import Flask, jsonify, request

from predict import run_prediction
from pipeline.logger import setup_logger

logger = setup_logger(__name__)

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    logger.info(f"API predict with payload {data}.")

    prediction = run_prediction(data)

    logger.info(f"API predictions {prediction}.")

    return jsonify({"data": prediction})


if __name__ == "__main__":
    app.run(threaded=True, port=5000)
