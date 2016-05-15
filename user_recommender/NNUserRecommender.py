""" Nearest-Neigbour-based User Recommendation System
Author: Yi Zhang <beingzy@gmail.com>
Date: 2016/02/20
"""
from numpy import array, vstack
from . import UserRecommenderMixin
from ..distance_metrics import GeneralDistanceWrapper
from .PairwiseDistMatrix import PairwiseDistMatrix


class NNUserRecommender(UserRecommenderMixin):

    def __init__(self, user_ids, user_profiles, user_connections, weights=None):
        super().__init__()
        # load user-related information
        self.load_user_ids(user_ids)
        self.load_user_profiles(user_profiles)
        self.load_user_connections(user_connections)

        # store ordered un-connected users
        # which would be reset per every
        # distance metrics update
        self._ordered_cand_dict = {}

        # load generalized distance wrapper to deal with cateogrical features
        self._gd_wrapper = GeneralDistanceWrapper()
        # learn which features of user_profile are categorical
        self._gd_wrapper.fit(self._user_profiles)
        # initiate default weights for distance caluclation
        self._gd_wrapper.load_weights(weights)

        # defulat maximum number of suggsetion per recommendation query
        self._size = 5

        # store ordered un-connected users
        # which would be reset per every
        # distance metrics update
        self._ordered_cand_dict = {}
        self._recommended_user_dict = {}

    def _update_dist_func(self):
        self._gd_wrapper.fit(self._user_profiles)
        fit_dist_func = self._gd_wrapper.dist_euclidean
        self._dist_matrix.set_dist_func(fit_dist_func)

    def set_recommendation_size(self, size=5):
        self._size = size

    def _get_distance(self, a_user_id, b_user_id):
        """ return distance between two user
        """
        a_user_idx = [i for i, uid in enumerate(self._user_ids) if uid == a_user_id][0]
        b_user_idx = [i for i, uid in enumerate(self._user_ids) if uid == b_user_id][0]
        a_user_profile = self._user_profiles[a_user_idx, :]
        b_user_profile = self._user_profiles[b_user_idx, :]
        return self._gd_wrapper.dist_euclidean(a_user_profile, b_user_profile)

    def update(self, **kwargs):
        """ update social network """
        if "user_ids" in kwargs.keys():
            self.load_user_ids(kwargs["user_ids"])

        if "user_profiles" in kwargs.keys():
            self.load_user_profiles(kwargs["user_profiles"])
            self._update_dist_func()

        if "user_connections" in kwargs.keys():
            self.load_user_connections(kwargs["user_connections"])

        # update distance matrix
        if "user_ids" in kwargs.keys() or "user_profiles" in kwargs.keys():
            # !!!! --- SHOULD REMOVE THIS LINE DUE TO UN_MATCHED DISTANCE WEIGHTS
            self._dist_matrix = PairwiseDistMatrix(self._user_ids, self._user_profiles)
            # self._dist_matrix.set_dist_func()
            self._dist_matrix.update_distance_matrix()

    def add_new_connections(self, new_user_connections):
        """ add new user connections """
        # user_connections = [[uid_a, uid_b], [uid_b, uid_c]]
        if isinstance(new_user_connections, list):
            if isinstance(new_user_connections[0], list):
                new_user_connections = array(new_user_connections)
            else:
                raise ValueError("illegal new_user_connections data is provided !")
        self._user_connections = vstack((self._user_connections, new_user_connections))

    def get_connected_users(self, user_id):
        """ return a list of user who are connceted with the target user"""
        b_user_ids = [pp[1] for pp in self._user_connections if pp[0] == user_id]
        a_user_ids = [pp[0] for pp in self._user_connections if pp[1] == user_id]
        return list(set(b_user_ids + a_user_ids))

    def gen_suggestion(self, user_id, block_list=[]):
        """ generate recommendation for a specified user """
        size = self._size

        if user_id in self._ordered_cand_dict:
            sorted_cand_uids = self._ordered_cand_dict[user_id]
            if len(sorted_cand_uids) > size:
                suggestion = sorted_cand_uids[:size]
                del self._ordered_cand_dict[user_id][:size]
            else:
                suggestion = sorted_cand_uids
                self._ordered_cand_dict[user_id] = []
            return suggestion

        else:
            cand_user_ids, cand_user_dist = self._dist_matrix.list_all_dist(user_id)
            con_user_ids = self.get_connected_users(user_id)
            if len(block_list) == 0:
                block_list = con_user_ids + [user_id]
            else:
                block_list.extend(con_user_ids + [user_id])

            # remove connected users from condidate list
            keep_idx = [ii for ii, cand_user_id in enumerate(cand_user_ids) if not cand_user_id in block_list]
            cand_user_ids = [cand_user_ids[ii] for ii in keep_idx]
            cand_user_dist = []
            for ii, cand_user_id in cand_user_ids:
                dist = self._get_distance(user_id, cand_user_id)
                cand_user_dist.append(dist)

            # sort candidates by distance
            sorted_list = sorted(zip(cand_user_ids, cand_user_dist), key=lambda pp: pp[1])
            sorted_cand_uids = [uid for uid, _ in sorted_list]

            suggestion = sorted_cand_uids[:size]
            del sorted_cand_uids[:size]
            self._ordered_cand_dict[user_id] = sorted_cand_uids.copy()

            return suggestion

    def update_reject_dict(self, user_id, rejected_list):
        if user_id in self._recommended_user_dict:
            self._recommended_user_dict[user_id].extend(rejected_list)
        else:
            self._recommended_user_dict[user_id] = rejected_list


class DNNUserRecommender(NNUserRecommender):
    """ Directed Network verion of NNUserRecommender
    """

    def __init__(self):
        super().__init__()

    def get_connected_users(self, user_id):
        user_ids = [pp[1] for pp in self._user_connections if pp[0] == user_id]
        return user_ids
