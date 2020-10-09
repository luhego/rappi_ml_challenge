import json

from logger import setup_logger

logger = setup_logger(__name__)


class AgeImputer:
    """Fills missing age rows with the Age median"""

    def _compute_age_median(self, df, train_mode):
        filepath = "../artifacts/age_median.json"
        if train_mode:
            age_median = df.Age.median()
            with open(filepath, "w") as file:
                json.dump({"age_median": age_median}, file)
        else:
            with open(filepath, "r") as file:
                age_median = json.load(file)["age_median"]
        return age_median

    def transform(self, df, train_mode):
        logger.info("Imputing missing Age values.")
        age_median = self._compute_age_median(df, train_mode)
        df.Age = df.Age.fillna(age_median)
        return df


class EmbarkedImputer:
    """Fills missing Embarked values with the Embarked mode."""

    def _compute_embarked_mode(self, df, train_mode):
        filepath = "../artifacts/embarked_mode.json"
        if train_mode:
            embarked_mode = df.Embarked.mode()[0]
            with open(filepath, "w") as file:
                json.dump({"embarked_mode": embarked_mode}, file)
        else:
            with open(filepath, "r") as file:
                embarked_mode = json.load(file)["embarked_mode"]
        return embarked_mode

    def transform(self, df, train_mode):
        logger.info("Imputing missing Embarked values.")
        embarked_mode = self._compute_embarked_mode(df, train_mode)
        df.Embarked = df.Embarked.fillna("S")
        return df


class DataImputer:
    def __init__(self, df):
        self.df = df
        self.imputers = [AgeImputer(), EmbarkedImputer()]

    def transform(self, train_mode=True):
        logger.info("Running DataImputer task.")
        df = self.df
        for imputer in self.imputers:
            df = imputer.transform(df, train_mode)
        return df
