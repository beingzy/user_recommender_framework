""" Nearest-Neigbour-based User Recommendation System
Author: Yi Zhang <beingzy@gmail.com>
Date: 2016/02/20
"""
from numpy import array, vstack
from user_recommender.UserRecommenderMixin import UserRecommenderMixin
from user_recommender.PairwiseDistMatrix import PairwiseDistMatrix

class NNUserRecommender(UserRecommenderMixin):

    def __init__(self, user_ids, user_profiles, user_connections):
        super().__init__()
        # load user-related information
        self.load_user_ids(user_ids)
        self.load_user_profiles(user_profiles)
        self.load_user_connections(user_connections)
        # create distance matrix
        self._dist_matrix = PairwiseDistMatrix(self._user_ids, self._user_profiles)
        self._dist_matrix.update_distance_matrix()
        # defulat maximum number of suggsetion per recommendation query
        self._size = 5

    def update(self, **kwargs):
        """ update social network """
        if "user_ids" in kwargs.keys():
            self.load_user_ids(kwargs["user_ids"])
        if "user_profiles" in kwargs.keys():
            self.load_user_profiles(kwargs["user_profiles"])
        if "user_connections" in kwargs.keys():
            self.load_user_connections(kwargs["user_connections"])

        # update distance matrix
        if "user_ids" in kwargs.keys() or "user_profiles" in kwargs.keys():
            self._dist_matrix = PairwiseDistMatrix(self._user_ids, self._user_profiles)
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

    def gen_suggestion(self, user_id):
        """ generate recommendation for a specified user """
        cand_user_ids, cand_user_dist = self._dist_matrix.list_all_dist(user_id)
        con_user_ids = self.get_connected_users(user_id)

        # remove connected users from condidate list
        keep_idx = [ii for ii, cand_user_id in enumerate(cand_user_ids) if not cand_user_id in con_user_ids]
        cand_user_ids = [cand_user_ids[ii] for ii in keep_idx]
        cand_user_dist = [cand_user_dist[ii] for ii in keep_idx]

        # sort candidates by distance
        sorted_list = sorted(zip(cand_user_ids, cand_user_dist), key=lambda pp: pp[1])
        sorted_cand_uids = [uid for uid, _ in sorted_list]

        size = self._size
        if len(sorted_cand_uids) > size:
            return sorted_cand_uids[:size]
        else:
            return sorted_cand_uids


class DNNUserRecommender(NNUserRecommender):
    """ Directed Network verion of NNUserRecommender
    """

    def __init__(self):
        super().__init__()

    def get_connected_users(self, user_id):
        user_ids = [pp[1] for pp in self._user_connections if pp[0] == user_id]
        return user_ids
