""" Simulation Experiment Framework for User Recommendation System
Author: Yi Zhang <beingzy@gmail.com>
Date: 2016/02/21
"""
from os import getcwd
from os.path import join
import warnings
from datetime import datetime
from tqdm import tqdm
from user_recommender import UserRecommenderMixin


class UserRecSysExpSimulator(object):
    def __init__(self, name=None, outpath=None):

        self.name = name
        if outpath is None:
            # log file export path
            self._outpath = getcwd()

        # experimentation state information
        self._iteration = 0

        # experiment ciritical components
        self._recommender = None
        self._clicker = None

        # experimentation information
        # initilize the experiment setting and can be modified anytime later
        # until the moment when the experiment starts
        self._set_info = {}
        self.set_recommendation_size()
        self.set_max_iterations()

    def load_init_data(self, user_ids, user_profiles, user_connections):
        """ load initial leanring data for experimentation """
        self._now_user_ids = user_ids
        self._now_user_profiles = user_profiles
        self._now_user_connections = user_connections

    def load_referrence_data(self, user_connections):
        """ load the referrence user connections """
        self._ref_user_connections = user_connections

    def load_recommender(self, recommender_class):
        """ load recommendation system class """
        self._recommender = recommender_class(user_ids=self._now_user_ids,
                                              user_profiles=self._now_user_profiles,
                                              user_connections=self._now_user_connections)
        self._recommender.set_recommendation_size(self._set_info["size"])

    def load_clicker(self, clicker_obj):
        """ load clikcer object """
        self._clicker = clicker_obj

    def load_evalutor(self, evalator=None):
        """ load evalutor compare current now_user_connections vs. ref_user_connections
        """
        self._evalutor = evalator

    def set_recommendation_size(self, size=None):
        """ set the size of suggestion per recommendation.
            it could not be modified after experimentation starts
        """
        if self._iteration == 0:
            # enable modification before experimentation starts
            if size is None:
                size = 5
            # update setting information
            self._set_info["size"] = size
            # update recommender setting
            if not self._recommender is None:
                self._recommender.set_recommendation_size(self._set_info["size"])
        else:
            raise EnvironmentError("setting update is forbidden after experiment started !")

    def set_max_iterations(self, value=None):
        """ set maximum number of iterations for experimtatnions
            it could not be modified after experimentation starts
        """
        if self._iteration == 0:
            if value is None:
                value = 100
            self._set_info["max_iter"] = value
        else:
            raise EnvironmentError("setting update is forbidden after experiment started !")

    def get_evalution(self):
        """ get evaluation score """
        return (0)

    def _update_one_step(self):
        """ experiment advance by one iteration """
        max_iter = self._set_info["max_iter"]
        if self._iteration < max_iter:
            start_time = datetime.now()

            # operation goes here ...
            #
            self._iteration += 1

            duration = datetime.now() - start_time
            total_cost = duration.total_seconds()
        else:
            msg = "experiment had reached the maximum iteration (%d)".format(max_iter)
            warnings.warn(msg)

    def start(self, display_progress=True):
        """ start experimentation until reach maximum itartion
        """
        start_time = datetime.now()
        max_iter = self._set_info["max_iter"]
        while self._iteration < max_iter:
            self._update_one_step()
        # export experiment information
        outfile = self.name + start_time.strftime("%Y%m%d_%H%M%S.csv")
        # write out test results
