from sklearn.decomposition import FactorAnalysis
import numpy as np
import matplotlib.pyplot as plt
import pandas
import random


class FA():

    def __init__(self):
        self.model_ = None
        self.components_ = None

    def _reset(self):
        self.model_ = None
        self.components_ = None

    def _fit(self, X, n_components, sample_size):
        self._reset()
        Y = np.delete(X, range(0,14), axis=1) #delete column numbers:1 included - 13 excluded, the knob columns are deleted
        np.random.shuffle(Y) #Shuffled only by the rows, by default
        Y=Y[:sample_size,:]#sample only 1000 rows from the matrix
        Y=Y.transpose()
        print("Shape before:", Y.shape)
        model= FactorAnalysis(n_components=n_components, random_state=0)
        model.fit_transform(Y)
        self.model_=model
        print(self.model_.components_.shape) #metrics X factors
        #filtering out any components with 0 values
        self.model_.components_=self.model_.components_.transpose()
        components_mask = np.sum(self.model_.components_ != 0.0, axis=1) > 0.0
        self.components_ = self.model_.components_[components_mask]
        print("Shape after:",self.components_.shape)

        return self
