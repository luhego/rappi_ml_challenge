import pandas as pd


class DataLoader:
    """Loads the dataset and returns a dataframe."""

    def __init__(self, filepath):
        self.filepath = filepath

    def load(self):
        df = pd.read_csv(self.filepath)
        return df
