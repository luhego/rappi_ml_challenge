import pickle
import random

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from logger import setup_logger

logger = setup_logger(__name__)


TEST_SIZE = 0.20
RANDOM_STATE = 23

# Set random seeds for reproducible results
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


class GenericModel:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def train(self):
        logger.info(f"Training model {self.name}.")
        self.model.fit(self.X_train, self.y_train)
        return self.model

    def evaluate(self):
        predictions = self.model.predict(self.X_test)
        score = accuracy_score(self.y_test, predictions)
        logger.info(f"Evaluation model {self.name}. Accuracy: {score}.")


class LinearModel(GenericModel):
    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(X_train, X_test, y_train, y_test)
        self.model = LinearSVC()
        self.name = "LinearSVC"
        self.filepath = "../artifacts/linsvc_clf.pkl"

    def persist(self):
        logger.info(f"Persisting model {self.name} in {self.filepath}.")
        with open(self.filepath, "wb") as file:
            pickle.dump(self.model, file)


class RandomForestModel(GenericModel):
    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(X_train, X_test, y_train, y_test)
        self.model = RandomForestClassifier()
        self.name = "RandomForestClassifier"
        self.filepath = "../artifacts/rf_clf.pkl"

    def persist(self):
        logger.info(f"Persisting model {self.name} in {self.filepath}.")
        with open(self.filepath, "wb") as file:
            pickle.dump(self.model, file)


class ModelTrainer:
    """Trains multiple models and store them for later usage."""

    def __init__(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        self.models = [
            LinearModel(X_train, X_test, y_train, y_test),
            RandomForestModel(X_train, X_test, y_train, y_test),
        ]

    def train(self):
        logger.info("Running ModelTrainer task.")
        for model in self.models:
            model.train()
            model.evaluate()

    def persist(self):
        logger.info("Persisting trained models.")
        for model in self.models:
            model.persist()
