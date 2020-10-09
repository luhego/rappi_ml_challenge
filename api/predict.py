import pandas as pd

from pipeline.data_cleaner import DataCleaner
from pipeline.data_imputer import DataImputer
from pipeline.feature_extractor import FeatureExtractor
from pipeline.model_predictor import LinearPredictor, RandomForestPredictor


def run_prediction(payload):
    """
    We process the payload and generate a prediction using the saved model.
    The payload has the following format:
        {
            "PassengerId": 1,
            "Pclass": 3,
            "Name": "Braund, Mr. Owen Harris",
            "Sex": "male",
            "Age": 22.0,
            "SibSp": 1,
            "Parch": 0,
            "Ticket": "A/5 21171",
            "Fare": 7.25,
            "Cabin": null,
            "Embarked": "S"
        }
    """

    # Convert the payload to a dataframe
    df = pd.Series(payload).to_frame().T

    # Run the pipeline
    df = DataCleaner(df).clean()
    df = DataImputer(df).transform(train_mode=False)
    df = FeatureExtractor(df).transform(train_mode=False)

    # Return the predictions
    return {
        "linsvc_prediction": int(LinearPredictor().predict(df)[0]),
        "rf_prediction": int(RandomForestPredictor().predict(df)[0]),
    }
