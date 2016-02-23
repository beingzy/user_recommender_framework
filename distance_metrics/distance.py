""" distance matrics
Author: Yi Zhang <beingzy@gmail.com>
Date: 2016/02/22
"""
from scipy.spatial.distance import euclidean


class DistanceMetrics(object):
    def __init__(self, feat_name=None):
        self._feat_names = feat_name
        self._cat_feat_names = None
        self._num_feat_names = None
        self._weights = None

    def set_categorical_features(self, cat_featnames=None, cat_idx=None):
        if not cat_featnames is None:
            if self._cat_feat_names is None:
                raise ValueError("_feat_names is not defined!")
            cat_idx = [ii for ii, featname in enumerate(self._cat_feat_names) \
                       if featname in cat_featnames]
        self._cat_idx = cat_idx
        self._num_idx = [ii for ii in range(len(self._feat_names)) \
                         if not ii in cat_idx]

    def _validate_data(self, x):
        """ validate supplied argument """
        if len(x) == len(self._feat_names):
            raise ValueError("x is incompatiable!")
