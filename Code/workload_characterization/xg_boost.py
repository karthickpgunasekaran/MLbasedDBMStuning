import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

class XGB(object):
    def __init__(self, lr, max_depth, alpha, n_estimators):
        self.model_ = xgb.XGBRegressor(objective='reg:linear', learning_rate=lr,
                                  max_depth=max_depth, alpha=alpha, n_estimators=n_estimators, eval_metric='map')
    def fit(self, X_train, Y_train):
        self.model_.fit(X_train, Y_train)
        return self

    def predict(self, X_test):
        predictions = self.model_.predict(X_test)
        return predictions










