import pandas as pd


class DummyCreator:
    def __init__(self, column, prefix=None):
        self.column = column
        self.prefix = prefix

    def tranform(self, df):
        column = self.column
        dummies = pd.get_dummies(df[column], prefix=self.prefix)
        df = df.drop([column], axis=1)
        df = df.join([dummies])
        return df


class FeatureExtractor:
    def __init__(self, df):
        self.df = df
        self.extractors = [
            DummyCreator('Embarked'),
            DummyCreator('Sex'),
            DummyCreator('Pclass', prefix="Class"),
        ]

    def transform(self):
        df = self.df
        for extractor in self.extractors:
            df = extractor.tranform(df)
        return df
