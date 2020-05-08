
from abc import ABCMeta, abstractproperty
from collections import OrderedDict
from sklearn.metrics import silhouette_score
import os
import json
import copy
import numpy as np
import scipy.stats
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal as mvn


class EM():

    SAMPLE_CUTOFF_ = 1000

    def __init__(self):
        self.model_ = None
        self.n_clusters_ = None
        self.sample_labels_ = None
        self.sample_distances_ = None

    @property
    def cluster_inertia_(self):
        # Sum of squared distances of samples to their closest cluster center
        return None if self.model_ is None else \
            self.model_.inertia_


    def cluster_labels_(self):
        # Cluster membership labels for each point
        return None if self.model_ is None else \
            copy.deepcopy(self.model_.labels_)


    def cluster_centers_(self):
        # Coordinates of the cluster centers
        return None if self.model_ is None else \
            copy.deepcopy(self.model_.cluster_centers_)

    def _reset(self):
        """Resets all attributes (erases the model)"""
        self.model_ = None
        self.n_clusters_ = None
        self.sample_labels_ = None
        self.sample_distances_ = None

    def fit(self, X, K, sample_labels=None, estimator_params=None):
        self._reset()
        self.model_=GaussianMixture(n_components=K,covariance_type='full', random_state=42).fit(X)
        model=self.model_
        if(model.converged_==False):
            print("Model has NOT converged")
        self.cluster_labels_=self.model_.predict(X)

        if sample_labels is None:
            print("INSIDE")
            sample_labels = ["sample_{}".format(i) for i in range(X.shape[0])]

        print("sample labels:",len(sample_labels)," ",X.shape[0])
        assert len(sample_labels) == X.shape[0]

        self.sample_labels_ = np.array(sample_labels)
        self.n_clusters_ = K

        # Record sample label/distance from its cluster center
        self.sample_distances_ = OrderedDict()
        for cluster_label in range(self.n_clusters_):
            assert cluster_label not in self.sample_distances_
            member_rows = X[self.cluster_labels_ == cluster_label, :]
            member_labels = self.sample_labels_[self.cluster_labels_ == cluster_label]

            self.cluster_centers_ = self.model_.means_
            centroid = np.expand_dims(self.cluster_centers_[cluster_label], axis=0)

            # "All clusters must have at least 1 member!"
            if member_rows.shape[0] == 0:
                return None

            # Calculate distance between each member row and the current cluster
            dists = np.empty(member_rows.shape[0])
            dist_labels = []
            for j, (row, label) in enumerate(zip(member_rows, member_labels)):
                dists[j] = cdist(np.expand_dims(row, axis=0), centroid, "euclidean").squeeze()
                dist_labels.append(label)

            # Sort the distances/labels in ascending order
            sort_order = np.argsort(dists)
            dists = dists[sort_order]
            dist_labels = np.array(dist_labels)[sort_order]
            self.sample_distances_[cluster_label] = {
                "sample_labels": dist_labels,
                "distances": dists,
            }
        return self


    def get_closest_samples(self):
        """Returns a list of the labels of the samples that are located closest
           to their cluster's center.
        Returns
        ----------
        closest_samples : list
                  A list of the sample labels that are located the closest to
                  their cluster's center.
        """
        if self.sample_distances_ is None:
            raise Exception("No model has been fit yet!")

        return [samples['sample_labels'][0] for samples in list(self.sample_distances_.values())]




class EMClusters():

    def __init__(self):
        self.min_cluster_ = None
        self.max_cluster_ = None
        self.cluster_map_ = None
        self.sample_labels_ = None




    def fit(self, X, min_cluster, max_cluster, sample_labels=None, estimator_params=None):

        self.min_cluster_ = None
        self.max_cluster_ = None
        self.cluster_map_ = None
        self.sample_labels_ = None
        self.min_cluster_ = min_cluster
        self.max_cluster_ = max_cluster
        self.cluster_map_ = {}
        #if sample_labels is None:
        #    sample_labels = ["sample_{}".format(i) for i in range(X.shape[1])]
        self.sample_labels_ = sample_labels
        for K in range(self.min_cluster_, self.max_cluster_ + 1):
            tmp = EM().fit(X, K, self.sample_labels_, estimator_params)
            if tmp is None:  # Set maximum cluster
                assert K > min_cluster, "min_cluster is too large for the model"
                self.max_cluster_ = K - 1
                break
            else:
                self.cluster_map_[K] = tmp

        return self
    def plot_graphs(self, X):
        n_components = np.arange(2, 21)
        models = [GaussianMixture(n, covariance_type='full').fit(X)
                  for n in n_components]
        plt.plot(n_components, [m.bic(X) for m in models], label='BIC')
        plt.legend(loc='Silhoutte score')
        plt.xlabel('# clusters')
        plt.ylabel('Silhoutte score')
        plt.show()

class Silhouette():
    """Det:

    Approximates the optimal number of clusters (K).


    References
    ----------
    http://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html


    Attributes
    ----------
    optimal_num_clusters_ : int
                            An estimation of the optimal number of clusters K for
                            KMeans models fit to X

    clusters_ : array, [n_clusters]
                The sizes of the clusters

    name_ : string
            The name of this technique

    Score_ : array, [n_clusters]
            The mean Silhouette Coefficient for each cluster size K
    """

    # short for Silhouette score
    NAME_ = "s-score"

    def __init__(self):
        super(Silhouette, self).__init__()
        self.scores_ = None

    @property
    def name_(self):
        return Silhouette.NAME_

    def _reset(self):
        """Resets all attributes (erases the model)"""
        self.optimal_num_clusters_ = None
        self.clusters_ = None
        self.scores_ = None

    def fit(self, X, cluster_map):
        """Estimates the optimal number of clusters (K) for a
           KMeans model trained on X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        cluster_map_ : dict
                       A dictionary mapping each cluster size (K) to the KMeans
                       model fitted to X with K clusters

        Returns
        -------
        self
        """
        self._reset()
        n_clusters = len(cluster_map)
        # scores = np.empty(n_clusters)
        scores = np.zeros(n_clusters)
        for i, (K, model) \
                in enumerate(sorted(cluster_map.items())):
            if K <= 1:  # K >= 2
                continue
            scores[i] = silhouette_score(X, model.cluster_labels_)

        self.clusters_ = np.array(sorted(cluster_map.keys()))
        self.optimal_num_clusters_ = self.clusters_[np.argmax(scores)]
        self.scores_ = scores
        return self

        # plt.plot(n_components, [m.bic(X) for m in models], label='BIC')


