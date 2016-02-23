""" distance matrics
Author: Yi Zhang <beingzy@gmail.com>
Date: 2016/02/22
"""
from numpy import sqrt
from pandas import DataFrame
from numpy import array, ndarray

class GeneralDistanceWrapper(object):
    """ Wrapper container to support generalized distance calculation
    for vectors involving both numeric and categorical values

    Example:
    --------
    # caluclate unweighted euclidean dsitance
    x, y =  [1, 2, 'a', 5.5, 'b'], [4, 0, 'a', 2.1, 'c']
    cat_dist_wrapper = GeneralDistanceWrapper()
    cat_dist_wrapper.update_category_index([2, 4])
    xydist = cat_dist_wrapper.dist_euclidean(x, y)

    # weighted euclidean distance
    weights = [1, 1, 1, 0, 0]
    cat_dist_wrapper.load_weights(weights)
    xydist_weighted = cat_dist_wrapper.dist_euclidean(x, y)

    # extract generalized distance calucaltion
    # dist_func is immutable, weights will be automatically
    # update in response to cat_dist_wrapper.load_weights()
    dist_func = cat_dist_wrapper.dist_euclidean
    xydist_weighted = dist_func(x, y)
    """

    def __init__(self, category_index=None):
        self.cat_idx = category_index
        self.load_weights()

    def fit(self, x):
        """ automate the detection of categoricay variables
        """
        category_dtypes = [str, bool, object]

        if isinstance(x, list):
            cat_idx = [ii for ii, val in enumerate(x) if type(val) in category_dtypes]
            self.cat_idx = cat_idx

        if isinstance(x, ndarray):
            first_row = x[0, :].tolist()
            cat_idx = [ii for ii, val in enumerate(first_row) if type(val) in category_dtypes]
            self.cat_idx = cat_idx

        if isinstance(x, DataFrame):
            all_feat_names = x.columns.tolist()
            first_row = x.iloc[0, :].tolist()
            cat_idx = [ii for ii, val in enumerate(first_row) if type(val) in category_dtypes]
            cat_feat_names = [feat_name for ii, feat_name in enumerate(all_feat_names) if ii in cat_idx]
            self.set_features(all_feat_names=all_feat_names, cat_feat_names=cat_feat_names)
            pass

    def set_features(self, all_feat_names, cat_feat_names):
        self._all_feat_names = all_feat_names
        self._cat_feat_names = cat_feat_names
        self.cat_idx = [ii for ii, feat in enumerate(all_feat_names) if feat in cat_feat_names]

    def load_weights(self, weights=None, normalize=False):
        if not weights is None:
            if normalize:
                # normalize weights
                sum_weights = sum(weights)
                weights = [w / sum_weights for w in weights]
        self._weights = weights

    def reset_weights(self):
        self._weights = None

    def update_category_index(self, category_index):
        self.cat_idx = category_index

    def decompose(self, x):
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
        a_num, a_cat = self.decompose(a)
        b_num, b_cat = self.decompose(b)
        num_diff = [a - b for a, b in zip(a_num, b_num)]
        cat_diff = [0 if a == b else 1 for a, b in zip(a_cat, b_cat)]
        return (num_diff, cat_diff)

    def get_difference(self, a, b):
        num_diff, cat_diff = self.get_component_difference(a, b)
        return self.recover_vector_from_components(num_diff, cat_diff)

    def dist_euclidean(self, a, b):
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
