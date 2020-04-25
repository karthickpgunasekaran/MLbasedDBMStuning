#
# OtterTune - cluster.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
from abc import ABCMeta, abstractproperty
from collections import OrderedDict

import os
import json
import copy
import numpy as np

from scipy.spatial.distance import cdist

from sklearn.cluster import KMeans as SklearnKMeans



class KMeans():

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

    @property
    def cluster_labels_(self):
        # Cluster membership labels for each point
        return None if self.model_ is None else \
            copy.deepcopy(self.model_.labels_)

    @property
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
        # Note: previously set n_init=50
        self.model_ = SklearnKMeans(K)
        if estimator_params is not None:
            assert isinstance(estimator_params, dict)
            self.model_.set_params(**estimator_params)

        # Compute Kmeans model
        self.model_.fit(X)
        if sample_labels is None:
            #print("INSIDE")
            sample_labels = ["sample_{}".format(i) for i in range(X.shape[0])]

        print("sample labels:",len(sample_labels)," ",X.shape)
        assert len(sample_labels) == X.shape[0]

        self.sample_labels_ = np.array(sample_labels)
        self.n_clusters_ = K

        # Record sample label/distance from its cluster center
        self.sample_distances_ = OrderedDict()
        for cluster_label in range(self.n_clusters_):
            assert cluster_label not in self.sample_distances_
            member_rows = X[self.cluster_labels_ == cluster_label, :]
            member_labels = self.sample_labels_[self.cluster_labels_ == cluster_label]
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




class KMeansClusters():

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
            tmp = KMeans().fit(X, K, self.sample_labels_, estimator_params)
            if tmp is None:  # Set maximum cluster
                assert K > min_cluster, "min_cluster is too large for the model"
                self.max_cluster_ = K - 1
                break
            else:
                self.cluster_map_[K] = tmp

        return self



class DetK:

    NAME_ = "det-k"

    def __init__(self):
        super(DetK, self).__init__()
        self.fs_ = None


    def fit(self, X, cluster_map):

        self.optimal_num_clusters_ = None
        self.clusters_ = None
        self.fs_ = None

        n_clusters = len(cluster_map)
        nd = X.shape[1]
        fs = np.empty(n_clusters)
        sks = np.empty(n_clusters)
        alpha = {}
        # K from 1 to maximum_cluster_
        for i, (K, model) \
                in enumerate(sorted(cluster_map.items())):
            # Compute alpha(K, nd) (i.e. alpha[K])
            if K == 2:
                alpha[K] = 1 - 3.0 / (4 * nd)
            elif K > 2:
                alpha[K] = alpha[K - 1] + (1 - alpha[K - 1]) / 6.0
            sks[i] = model.cluster_inertia_

            if K == 1:
                fs[i] = 1
            elif sks[i - 1] == 0:
                fs[i] = 1
            else:
                fs[i] = sks[i] / (alpha[K] * sks[i - 1])
        self.clusters_ = np.array(sorted(cluster_map.keys()))
        self.optimal_num_clusters_ = self.clusters_[np.argmin(fs)]
        self.fs_ = fs
        return self


