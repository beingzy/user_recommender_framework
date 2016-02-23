""" distance matrics
Author: Yi Zhang <beingzy@gmail.com>
Date: 2016/02/22
"""
from scipy.spatial.distance import euclidean
from numpy import array
from numpy import sqrt

class CategoryWrapper(object):
    def __init__(self, category_index=None):
        self.cat_idx = category_index
        self.load_weights()

    def set_features(self, all_feat_names, cat_feat_names):
        self._all_feat_names = all_feat_names
        self._cat_feat_names = cat_feat_names
        self.cat_idx = [ii for ii, feat in enumerate(all_feat_names) if feat in cat_feat_names]

    def load_weights(self, weights=None):
        self._weights = weights

    def reset_weights(self):
        self._weights = None

    def update_category_index(self, category_index):
        self.cat_idx = category_index

    def wrapper(self, x):
        num_elements = [val for ii, val in enumerate(x) if not ii in self.cat_idx]
        cat_elements = [val for ii, val in enumerate(x) if ii in self.cat_idx]
        return (num_elements, cat_elements)

    def recover_vector_from_components(self, num_component, cat_component):
        # output conatainer
        tot_elements = len(num_component) + len(cat_component)
        cat_idx = self.cat_idx
        num_idx = [ii for ii in range(tot_elements) if not ii in cat_idx]
        vector = [None] * tot_elements
        for idx, val in zip(cat_idx, cat_component):
            vector[idx] = val
        for idx, val in zip(num_idx, num_component):
            vector[idx] = val
        return vector

    def get_component_difference(self, a, b):
        if len(a) != len(b):
            raise ValueError("vector (a) is in different size of vector (b)!")
        a_num, a_cat = self.wrapper(a)
        b_num, b_cat = self.wrapper(b)
        num_diff = [a - b for a, b in zip(a_num, b_num)]
        cat_diff = [0 if a == b else 1 for a, b in zip(a_cat, b_cat)]
        return (num_diff, cat_diff)

    def get_difference(self, a, b):
        num_diff, cat_diff = self.get_component_difference(a, b)
        return self.recover_vector_from_components(num_diff, cat_diff)

    def cal_euclidean(self, a, b):
        """ calculate the weighted euclidean distance
        """
        diff = self.get_difference(a, b)
        if self._weights is None:
            return sqrt(sum([val * val for val in diff]))
        else:
            if len(self._weights) == len(diff):
                return sqrt(sum([w * val * val for w, val in zip(self._weights, diff)]))
            else:
                raise ValueError("weights must be in same size with input vector (a, b)!")
