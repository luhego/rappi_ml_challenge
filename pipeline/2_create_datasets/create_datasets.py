from pathlib import Path

from sklearn.model_selection import train_test_split
import pandas as pd

TEST_SIZE = 0.20
RANDOM_STATE = 23


def _save_datasets(train_df, test_df, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir()

    out_train = out_dir / "train.csv"
    out_test = out_dir / "test.csv"


    train_df.to_csv(out_train, index=False)
    test_df.to_csv(out_test, index=False)


def create_datasets(in_csv, out_dir):
    """We split that main dataset into two datasets: train and test and store them in disk."""
    df = pd.read_csv(in_csv)

    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    _save_datasets(train_df, test_df, out_dir)


if __name__ == "__main__":
    create_datasets(in_csv="data/train.csv", out_dir="data/2")
