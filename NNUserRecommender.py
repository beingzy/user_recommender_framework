""" Nearest-Neigbour-based User Recommendation System
Author: Yi Zhang <beingzy@gmail.com>
Date: 2016/02/20
"""
from UserRecommender import UserRecommenderMixin

class NNUserRecommender(UserRecommenderMixin):

    def update_distance_matrix(self):
        """calcuate the distance matrix for user-pair"""
        pass
