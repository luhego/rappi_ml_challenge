import pandas as pd


class DataLoader:
    """It loads the dataset and return a pandas dataframe."""

    def __init__(self, filepath):
        self.filepath = filepath

    def load(self):
        df = pd.read_csv(self.filepath)
        return df
