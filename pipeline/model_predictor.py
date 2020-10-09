import pickle


class GenericPredictor:
    def predict(self, X):
        with open(self.filepath, "rb") as file:
            loaded_model = pickle.load(file)
            return loaded_model.predict(X)


class LinearPredictor(GenericPredictor):
    def __init__(self):
        self.filepath = "../artifacts/linsvc_clf.pkl"


class RandomForestPredictor(GenericPredictor):
    def __init__(self):
        self.filepath = "../artifacts/rf_clf.pkl"
