from pathlib import Path

import pandas as pd


def _save_datasets(train_df, test_df, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir()

    out_train = out_dir / "train.csv"
    out_test = out_dir / "test.csv"


    train_df.to_csv(out_train, index=False)
    test_df.to_csv(out_test, index=False)



def _remove_unused_cols(df):
    df = df.drop(['PassengerId','Name','Ticket', 'Cabin'], axis=1)
    return df


def clean_data(in_train_csv, in_test_csv, out_dir):
    train_df = pd.read_csv(in_train_csv)
    test_df = pd.read_csv(in_test_csv)

    cleaned_train_df = _remove_unused_cols(train_df)
    cleaned_test_df = _remove_unused_cols(test_df)

    _save_datasets(cleaned_train_df, cleaned_test_df, out_dir)

if __name__ == "__main__":
    clean_data(in_train_csv="data/2/train.csv", in_test_csv="data/2/test.csv", out_dir="data/3")
