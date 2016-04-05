""" User Recommender with Groupwise Distance Learning

Author: Yi Zhang <beingzy@gmail.com>
Date: 2016/04/05
"""
from user_recommender import UserRecommenderMixin
from distance_metrics import GeneralDistanceWrapper
from user_recommender.PairwiseDistMatrix import PairwiseDistMatrix
from groupwise_distance_learning.groupwise_distance_learner import GroupwiseDistLearner


class GWDUserRecommender(UserRecommenderMixin):

    def __init__(self, user_ids, user_profiles, user_connections):
        super().__init__()
        # load user-related information
        self.load_user_ids(user_ids)
        self.load_user_profiles(user_profiles)
        self.load_user_connections(user_connections)

        # equip with GWD leanner


    def _update_dist_func(self):
        pass

    def update(self, **kwargs):
        """ update social network
        """
        pass

    def add_new_connections(self, new_user_connections):
        pass

    def get_connected_users(self, user_ids):
        pass

    def gen_suggestion(self, user_id):
        pass