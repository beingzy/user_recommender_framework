""" Mixin Class to define the common methods for User Recommendation System methods.
Author: Yi Zhang <beingzy@gmail.com>
Date: 2016/02/20
"""

class UserRecommenderMixin(object):

    def __init__(self, **kwargs):
        if "user_ids" in kwargs.keys():
            self.load_user_ids(kwargs["user_ids"])
        else:
            self._user_ids = None

        if "user_profiles" in kwargs.keys():
            self.load_user_profiles(kwargs["user_profiles"])
        else:
            self._user_profiles = None

        if "user_connections" in kwargs.keys():
            self.load_user_connections(kwargs["user_connections"])
        else:
            self._user_connections = None
        # the recomemndation size (the number of suggestions per query)
        self._size = 5

        # properties will be accessed by experiment simulator
        # to record the advancement of experiment
        self._iter_counter = 0

    def set_recommendation_size(self, value):
        """ set the number of suggestions per recommendation """
        self._size = value

    def load_user_ids(self, value):
        """load a list of unque user ids"""
        self._user_ids = value

    def load_user_profiles(self, value):
        """load matrix-like user-profile (ordered by user id)"""
        self._user_profiles = value

    def load_user_connections(self, value):
        """load pair-wise connection"""
        self._user_connections = value

    def update_iteration(self):
        self._iter_counter += 1

    def update(self):
        pass

    def gen_suggestion(self, user_id):
        pass
