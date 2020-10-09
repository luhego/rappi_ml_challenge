class DataCleaner:
    """Cleans the received dataframe."""

    def __init__(self, df):
        self.df = df

    def clean(self):
        # Remove unused columns
        cleaned_df = self.df.drop(
            ['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
        return cleaned_df
