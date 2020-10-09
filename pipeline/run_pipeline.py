from data_loader import DataLoader
from data_cleaner import DataCleaner
from data_imputer import DataImputer
from feature_extractor import FeatureExtractor


def init():
    df = DataLoader("../data/train.csv").load()
    df = DataCleaner(df).clean()
    df = DataImputer(df).transform()
    df = FeatureExtractor(df).transform()
    print(df.head())


if __name__ == "__main__":
    init()
