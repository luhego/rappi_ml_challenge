import pandas as pd

from logger import logger


class DataLoader:
    """Loads the dataset and returns a dataframe."""

    def __init__(self, filepath):
        self.filepath = filepath

    def load(self):
        logger.info("Running data loading task.")

        df = pd.read_csv(self.filepath)
        return df
