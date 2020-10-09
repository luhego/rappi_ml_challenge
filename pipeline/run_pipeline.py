from data_loader import DataLoader
from data_cleaner import DataCleaner
from data_imputer import DataImputer
from feature_extractor import FeatureExtractor
from model_trainer import ModelTrainer


def init():
    df = DataLoader("../data/train.csv").load()
    df = DataCleaner(df).clean()
    df = DataImputer(df).transform()
    df = FeatureExtractor(df).transform()

    X = df.drop("Survived", axis=1)
    y = df.Survived

    mt = ModelTrainer(X, y)
    mt.train()
    mt.persist()


if __name__ == "__main__":
    init()
