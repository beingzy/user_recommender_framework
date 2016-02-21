""" Mixin Class to define the common methods for User Recommendation System methods.
Author: Yi Zhang <beingzy@gmail.com>
Date: 2016/02/20
"""

class UserRecommenderMixin(object):

    def __init__(self):
        self._user_ids = None
        self._user_profiles = None
        self._user_connections = None

    def load_user_ids(self, value):
        """load a list of unque user ids"""
        self._user_ids = value

    def load_user_profiles(self, value):
        """load matrix-like user-profile (ordered by user id)"""
        self._user_profiles = value

    def load_user_connections(self, value):
        """load pair-wise connection"""
        self._user_connections = value

    def update(self):
        pass

    def gen_suggestion(self, user_id, size=5):
        pass
