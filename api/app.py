from flask import Flask, jsonify, request

from error_handlers import InvalidUsage
from validators import PayloadValidator
from predict import run_prediction
from pipeline.logger import setup_logger

logger = setup_logger(__name__)

app = Flask(__name__)


@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if not data:
        raise InvalidUsage("The payload is empty.", status_code=400)

    validator = PayloadValidator(data)
    if not validator.is_valid():
        raise InvalidUsage(validator.error_message, status_code=400)

    logger.info(f"API predict with payload {data}.")

    prediction = run_prediction(data)

    logger.info(f"API predictions {prediction}.")

    return jsonify({"data": prediction})


if __name__ == "__main__":
    app.run(threaded=True, port=5000)
