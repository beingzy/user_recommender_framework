""" Pair-wise Distance Matrix Container
Author: Yi Zhang <beingzy@gmail.com>
Date: 2016/02/20
"""
from pandas import DataFrame
from itertools import combinations
# from scipy.spatial.distance import euclidean
from distance_metrics import GeneralDistanceWrapper

class PairwiseDistMatrix(object):

    def __init__(self, user_ids, user_profiles, target_user_ids=None):
        """ instance initilization function

        Parameters:
        ===========
        user_ids: <list> a list of unique user ids
        user_profiles: <matrix-like> a numpy.array of user profile
        target_uer_ids: <list> a list of user whose distance in relation to other users
            will be  calculated only.
        """
        if isinstance(user_profiles, DataFrame):
            user_profiles = user_profiles.as_matrix()

        if len(user_ids) != user_profiles.shape[0]:
            nobs = len(user_ids)
            nrow = user_profiles.shape[0]
            raise ValueError("user_ids (nobs: %d) does not match user_profiles (nrow: %d)".format(nobs, nrow))

        # distance wrapper
        dist_wrapper = GeneralDistanceWrapper()
        dist_wrapper.fit(user_profiles)
        self.set_dist_func(dist_wrapper.dist_euclidean)
        # state informaiton to track update
        self._update_status = True
        # user-related information
        self._user_ids = user_ids
        self._user_profiles = user_profiles
        self._target_user_ids = target_user_ids
        # user-pair-wise distance related info.
        self._dist_user_pairs = []
        self._dist_matrix = []

        self._all_pairs = []
        if target_user_ids is None:
            self._all_pairs = combinations(user_ids, 2)
        else:
            for a_uid in target_user_ids:
                for b_uid in user_ids:
                    if a_uid != b_uid:
                        self._all_pairs.append((a_uid, b_uid))

    def set_dist_func(self, dist_func):
        """ redefine the distance metric function """
        self._dist_func = dist_func

    def update_distance_matrix(self):
        """ update the distance matrix """
        # is_update = function() change
        all_pairs = self._all_pairs
        dist_user_pairs = []
        dist_containers = []
        for user_a, user_b in all_pairs:
            idx_a, idx_b = self._user_ids.index(user_a), self._user_ids.index(user_b)
            profile_a, profile_b = self._user_profiles[idx_a, :], self._user_profiles[idx_b, :]
            pair_dist = self._dist_func( profile_a, profile_b )
            dist_user_pairs.append( [user_a, user_b] )
            dist_containers.append( pair_dist )

        self._dist_user_pairs = dist_user_pairs
        self._dist_matrix = dist_containers
        # swith update status after refreshing distance matrix
        self._update_status = False

    def list_all_dist(self, user_id):
        """list distance involving user (user_id)

        Parameters:
        ----------
        * user_id: <string or integer>

        Returns:
        -------
        * (user_ids <list>, distance <list>)
        """
        if not user_id in self._user_ids:
            raise ValueError("user_id (:{}) is not included in data!".format(user_id))

        if not self._target_user_ids is None:
            if not user_id in self._target_user_ids:
                raise ValueError("user_id (:{})is not in target_user_ids!".format(user_id))

        # collect index of user-pair whose first user is target user
        idx_b = [ ii for ii, pp in enumerate(self._dist_user_pairs) if pp[0]==user_id ]
        user_id_b = [ pp[1] for pp in self._dist_user_pairs if pp[0]==user_id ]

        if not self._target_user_ids is None:
            # collect index of user-pair whose second user is target user
            idx_a = [ ii for ii, pp in enumerate(self._dist_user_pairs) if pp[1]==user_id ]
            user_id_a = [ pp[0] for pp in self._dist_user_pairs if pp[1]==user_id ]
        else:
            idx_a = []
            user_id_a = []

        # counter part user_ids
        cnu_ids = user_id_a + user_id_b
        idx = idx_a + idx_b
        pair_dist = [ self._dist_matrix[ii] for ii in idx ]
        return cnu_ids, pair_dist

