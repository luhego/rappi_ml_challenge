import pickle

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from logger import setup_logger

logger = setup_logger(__name__)


class OneHotEncoderTransformer:
    """Generates dummy columns for a given column. The original column is dropped."""

    def __init__(self, column):
        self.column = column

    def tranform(self, df, train_mode):
        logger.info(f"Generating dummy variables for column {self.column}.")
        column = self.column

        filepath = f"../artifacts/ohe_{column}.pkl"
        if train_mode:
            ohe_encoder = OneHotEncoder()
            X = ohe_encoder.fit_transform(df[column].values.reshape(-1, 1)).toarray()
            with open(filepath, "wb") as file:
                pickle.dump(ohe_encoder, file)
        else:
            with open(filepath, "rb") as file:
                ohe_encoder = pickle.load(file)
                X = ohe_encoder.transform(df[column].values.reshape(-1, 1)).toarray()

        ohe_df = pd.DataFrame(X, columns=[f"{column}_{i}" for i in range(X.shape[1])])

        df = df.drop([column], axis=1)
        df = pd.concat([df, ohe_df], axis=1)

        return df


class FeatureExtractor:
    """Extracts additional features for the training dataframe."""

    def __init__(self, df):
        self.df = df
        self.extractors = [
            OneHotEncoderTransformer("Embarked"),
            OneHotEncoderTransformer("Sex"),
            OneHotEncoderTransformer("Pclass"),
        ]

    def transform(self, train_mode=True):
        logger.info("Running Feature Extractor task.")
        df = self.df
        for extractor in self.extractors:
            df = extractor.tranform(df, train_mode)
        return df
