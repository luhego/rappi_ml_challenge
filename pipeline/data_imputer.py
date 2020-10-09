from logger import logger


class AgeImputer:
    """Fills missing age rows with the Age median"""

    def transform(self, df):
        logger.info("Imputing missing Age values.")
        df.Age = df.Age.fillna(df.Age.median())
        return df


class EmbarkedImputer:
    """Fills missing Embarked values with the Embarked mode."""

    def transform(self, df):
        logger.info("Imputing missing Embarked values.")
        df.Embarked = df.Embarked.fillna("S")
        return df


class DataImputer:
    def __init__(self, df):
        self.df = df
        self.imputers = [AgeImputer(), EmbarkedImputer()]

    def transform(self):
        logger.info("Running DataImputer task.")
        df = self.df
        for imputer in self.imputers:
            df = imputer.transform(df)
        return df
