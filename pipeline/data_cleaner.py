from logger import logger


class DataCleaner:
    """Cleans the received dataframe."""

    def __init__(self, df):
        self.df = df

    def clean(self):
        logger.info("Running data cleaning task.")

        # Remove unused columns
        cleaned_df = self.df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)
        return cleaned_df
