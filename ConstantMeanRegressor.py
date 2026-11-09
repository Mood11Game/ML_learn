import numpy as np

class ConstantMeanRegressor:
    def __init__(self):
        self.mean_value_ = None

    def fit(self, X, y):
        self.mean_value_ = y.mean()
        return self

    def predict(self, X):
        return np.full(shape=len(X), fill_value=self.mean_value_)

    
