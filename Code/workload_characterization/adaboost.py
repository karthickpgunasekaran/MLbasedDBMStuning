import numpy as np
import matplotlib.pyplot as plt
from gpr_model import GPRNP
from sklearn.ensemble import AdaBoostRegressor

class Adaboost(object):
    def __init__(self, lr, n_estimators):
        self.model_ = AdaBoostRegressor(GPRNP(),n_estimators=n_estimators, learning_rate=lr)

    def fit(self, X_train, Y_train):
        self.model_.fit(X_train, Y_train)
        return self

    def predict(self, X_test):
        predictions = self.model_.predict(X_test)
        return predictions

