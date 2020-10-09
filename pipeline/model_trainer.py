import pickle
import random

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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
        self.model.fit(self.X_train, self.y_train)
        return self.model

    def evaluate(self):
        predictions = self.model.predict(self.X_test)
        score = accuracy_score(self.y_test, predictions)
        print(score)


class LinearModel(GenericModel):
    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(X_train, X_test, y_train, y_test)
        self.model = LinearSVC()

    def persist(self):
        with open("../models/linsvc_clf.pkl", "wb") as file:
            pickle.dump(self.model, file)


class RandomForestModel(GenericModel):
    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(X_train, X_test, y_train, y_test)
        self.model = RandomForestClassifier()

    def persist(self):
        with open("../models/rf_clf.pkl", "wb") as file:
            pickle.dump(self.model, file)


class ModelTrainer:
    """Trains multiple models and store them for later usage."""

    def __init__(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        self.models = [
            LinearModel(X_train, X_test, y_train, y_test),
            RandomForestModel(X_train, X_test, y_train, y_test)
        ]

    def train(self):
        for model in self.models:
            model.train()
            model.evaluate()

    def persist(self):
        for model in self.models:
            model.persist()
