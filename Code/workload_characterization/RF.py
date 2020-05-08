from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

class RF(object):
    def __init__(self, lr, max_depth, n_estimators, max_features):
        self.model_ = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators, max_features=max_features, random_state=0, criterion="mae")
    def fit(self, X_train, Y_train):
        self.model_.fit(X_train, Y_train.ravel())
        return self

    def predict(self, X_test):
        predictions = self.model_.predict(X_test)
        return predictions










