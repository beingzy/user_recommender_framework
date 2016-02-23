""" Simulation Experiment Framework for User Recommendation System
Author: Yi Zhang <beingzy@gmail.com>
Date: 2016/02/21
"""
import os
from os import getcwd
from os.path import join, exists
import warnings
from datetime import datetime
import numpy as np
from numpy import array
from pandas import DataFrame
from tqdm import tqdm
from user_recommender.UserRecommenderMixin import UserRecommenderMixin
from network_simulator.SocialNetworkEvaluator import EvaluatorMixin


class UserRecSysExpSimulator(object):

    def __init__(self, name=None, outpath=None, is_directed=False):

        self.name = name
        if outpath is None:
            # log file export path
            self._outpath = join(getcwd(), "log")
            if not exists(self._outpath):
                os.mkdir(self._outpath)
        else:
            self._outpath = outpath
            if not exists(self._outpath):
                os.mkdir(self._outpath)

        # set network directed status
        self._is_directed = is_directed

        # experimentation state information
        self._iteration = 0

        # experiment ciritical components
        self._recommender = None
        self._clicker = None
        self._evaluator = None

        # data batch
        self._now_user_ids = None
        self._now_user_profiles = None
        self._now_user_connections = None
        self._ref_user_connections = None

        # experimentation information
        # initilize the experiment setting and can be modified anytime later
        # until the moment when the experiment starts
        self._set_info = {}
        self.set_recommendation_size()
        self.set_max_iterations()

    def load_now_user_ids(self, user_ids):
        if isinstance(user_ids, list):
            self._now_user_ids = user_ids
        else:
            raise ValueError("user_ids is not list.")

    def laod_now_user_profiles(self, user_profiles):
        if isinstance(user_profiles, np.ndarray):
            self._now_user_profiles = user_profiles
        else:
            raise ValueError("user_profiles is not numpy.array object.")

    def load_now_user_connections(self, user_connections):
        if isinstance(user_connections, np.ndarray):
            self._now_user_connections = user_connections
        else:
            raise ValueError("user_connections is not numpy.array object.")

    def load_init_data(self, user_ids, user_profiles, user_connections):
        """ load initial learning data for experimentation """
        self.load_now_user_ids(user_ids)
        self.laod_now_user_profiles(user_profiles)
        self.load_now_user_connections(user_connections)

    def load_referrence_data(self, user_connections):
        """ load the referrence user connections """
        self._ref_user_connections = user_connections

    def load_recommender(self, recommender_class):
        """ load recommendation system class """
        if issubclass(recommender_class, UserRecommenderMixin):
            self._recommender = recommender_class(user_ids=self._now_user_ids,
                                                  user_profiles=self._now_user_profiles,
                                                  user_connections=self._now_user_connections)
            self._recommender.set_recommendation_size(self._set_info["size"])
        else:
            raise ValueError("supplied recommender_class does not meet the requirement (UserRecommenderMixin) !")

    def load_clicker(self, clicker_class):
        """ load clikcer object """
        self._clicker = clicker_class()

    def load_evaluator(self, evaluator_class):
        """ load evalutor compare current now_user_connections vs. ref_user_connections
        """
        if issubclass(evaluator_class, EvaluatorMixin):
            self._evaluator = evaluator_class(is_directed=self._is_directed)
            self._evaluator.load_ref_user_connections(self._ref_user_connections)
            self._evaluator.load_eval_user_connections(self._now_user_connections)
        else:
            raise ValueError("supplied evalutor_class does not meet the requirement (sub-class of EvaluatorMixin) !")

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

    def _validation_information(self):
        """ self-check if all information required for experiment is ready """
        if self._recommender is None:
            raise ValueError("recommender is not defined yet! user .load_recommender(cls) method to define.")
        if self._evaluator is None:
            raise ValueError("evaluator is not defined yet! use .load_evalutor(cls) method to define.")
        if self._clicker is None:
            raise ValueError("clicker is not defined yet! user .load_clicker(cls) method to define.")

        if self._now_user_ids is None:
            raise ValueError("user_ids is not defined!")
        if self._now_user_profiles is None:
            raise ValueError("now_user_profiles is not defined!")
        if self._now_user_connections is None:
            raise ValueError("now_user_connections is not defined!")
        if self._ref_user_connections is None:
            raise ValueError("ref_user_connections is not defined!")

        pass

    def _update_one_step(self):
        """ experiment advance by one iteration """
        max_iter = self._set_info["max_iter"]

        new_connections = []
        if self._iteration < max_iter:
            start_time = datetime.now()
            # operation goes here ...
            uniq_user_ids = self._now_user_ids
            for ii, user_id in enumerate(uniq_user_ids):
                suggestions = self._recommender.gen_suggestion(user_id=user_id)
                confirms = self._clicker.click(suggestions)
                print("---- number of suggestions: " + str(len(suggestions)))
                if len(confirms) > 0:
                    pairs = [[user_id, confirm] for confirm in confirms]
                    if len(new_connections) == 0:
                        new_connections = pairs
                    else:
                        new_connections.extend(pairs)
            # consolidate new connections
            new_connections = array(new_connections)

            # tracking experiment progress
            self._iteration += 1

            if new_connections.shape[0] > 0:
                updated_user_connections = np.vstack((self._now_user_connections, new_connections))
                # update simulator's connection data
                # self.load_now_user_connections(updated_user_connections)
                self._recommender.load_user_connections(updated_user_connections)
            else:
                msg = str(self._iteration) + " iteration: no new connections are created !"
                warnings.warn(msg)

            duration = datetime.now() - start_time
            total_cost = duration.total_seconds()

            # collect evaluation scores
            self._evaluator.load_eval_user_connections(self._recommender._user_connections)
            eval_score = self._evaluator.get_score()

            # collect information
            exp_record = {"iteration": self._iteration,
                          "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                          "time_cost_seconds": total_cost,
                          "num_new_connections": new_connections.shape[0]}
            exp_record.update(eval_score)

            return exp_record

        else:
            msg = "experiment had reached the maximum iteration (max: " + str(max_iter) + ")"
            warnings.warn(msg)

    def run(self):
        """ start experimentation until reach maximum itartion
        """
        # check if all information is ready
        self._validation_information()

        start_time = datetime.now()
        max_iter = self._set_info["max_iter"]

        exp_records = []
        with tqdm(total=max_iter) as pbar:
            for ii in tqdm(range(max_iter)):
                record = self._update_one_step()
                exp_records.append(record)
                pbar.update()

        # export experiment information
        fname = self.name + start_time.strftime("_%Y%m%d_%H%M%S.csv")
        outfile = join(self._outpath, fname)
        # write out test results
        DataFrame(exp_records).to_csv(outfile, header=True, index=False)
