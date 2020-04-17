from sklearn.decomposition import FactorAnalysis
import numpy as np
import pandas
import random


class FA():

    def __init__(self):
        self.model_ = None
        self.components_ = None

    def _reset(self):
        self.model_ = None
        self.components_ = None

    def _fit(X, n_components, sample_size):
        self._reset()
        X = np.delete(X, range(1,13), axis=1) #delete column numbers:1 included - 13 excluded, the knob columns are deleted
        np.random.shuffle(X) #Shuffled only by the rows, by default
        X=X[:sample_size,:]#sample only 1000 rows from the matrix
        print(X.shape)
        self.model_ = FactorAnalysis(n_components=n_components, random_state=0)
        self.model_.fit_transform(X)
        print(self.model_.components_.shape) #metrics X factors

        #filtering out any components with 0 values
        components_mask = np.sum(self.model_.components_ != 0.0, axis=1) > 0.0
        self.components_ = self.model_.components_[components_mask]
        print(self.components_.shape)
        return self
