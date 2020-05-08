from keras.wrappers.scikit_learn import KerasRegressor
from keras import Sequential
from keras.layers import Dense
import pandas as pd

from Code.workload_characterization.gpr_model import GPRResult


class NN():
    def __init__(self):
        self.model_= KerasRegressor(build_fn=self.build_regressor, batch_size=32, epochs=100)

    def build_regressor(self):
        regressor = Sequential()
        regressor.add(Dense(units=50, input_dim=12))
        regressor.add(Dense(units=1))
        regressor.compile(optimizer='adam', loss='mean_absolute_percentage_error',
                          metrics=['mean_absolute_percentage_error'])
        return regressor
    def fit(self, X_workload, y_col):
        X_workload = pd.DataFrame(X_workload).iloc[:, 0:12]
        self.model_.fit(X_workload, y_col.ravel())
        return self

    def predict(self, X_target):
        return GPRResult(self.model_.predict(X_target), [])


