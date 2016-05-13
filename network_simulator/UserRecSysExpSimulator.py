""" Simulation Experiment Framework for User Recommendation System
Author: Yi Zhang <beingzy@gmail.com>
Date: 2016/05/02
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
from networkx import DiGraph, Graph
from .SocialNetworkEvaluator import EvaluatorMixin
from .UserClickSimulator import UserClickSimulatorMixin
from ..user_recommender.UserRecommenderMixin import UserRecommenderMixin
import copy


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
        self._no_growth_counter = 0
        self.set_no_growth_max(10)

        # experiment ciritical components
        self._recommender = None
        self._clicker = None
        self._evaluator = None

        # data batch
        self._init_user_connections = None
        self._ref_user_connections = None

        # experimentation information
        # initilize the experiment setting and can be modified anytime later
        # until the moment when the experiment starts
        self._set_info = {}
        self.set_recommendation_size()
        self.set_max_iterations()

        # store rejected users
        self._rejected_user_dict = {}

    def load_init_user_connections(self, user_connections):
        if isinstance(user_connections, np.ndarray):
            self._init_user_connections = user_connections
        else:
            raise ValueError("user_connections is not numpy.array object.")

    def load_ref_user_connections(self, user_connections):
        """ load the referrence user connections """
        self._ref_user_connections = user_connections

    def load_recommender(self, recommender_object):
        """ load recommendation system class """
        if isinstance(recommender_object, UserRecommenderMixin):
            self._recommender = recommender_object
            self._recommender.set_recommendation_size(self._set_info["size"])
            # attach data to simulator container
            if recommender_object._user_connections is None:
                raise ValueError("recommender's user_connections is not defined!")
            else:
                self.load_init_user_connections(recommender_object._user_connections)
        else:
            raise ValueError("the recommender_object is not an instance of UserRecommenderMixin or its child class!")

    def load_clicker(self, clicker_object):
        """ load clikcer object """
        if isinstance(clicker_object, UserClickSimulatorMixin):
            self._clicker = clicker_object
        else:
            raise ValueError("clicker_object is not an instance of UserClickSimulatorMixin or its child class!")

    def load_evaluator(self, evaluator_object):
        """ load evalutor compare current now_user_connections vs. ref_user_connections
        """
        if isinstance(evaluator_object, EvaluatorMixin):
            self._evaluator = evaluator_object
            # attach data to simulator container
            if evaluator_object._ref_user_connections is None:
                raise ValueError("evaluator's ref_user_connections is not defined!")
            else:
                self.load_ref_user_connections(evaluator_object._ref_user_connections)
        else:
            raise ValueError("evalutor_object is not an instance of EvaluatorMixin or its child class!")

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

    def set_no_growth_max(self, size=None):
        """ set upper limit for no growth iteration (connections)
            and stop experiment once the limit is reached
        """
        if size is None:
            size = 10
        self._no_growth_max = size


    def _validation_information(self):
        """ self-check if all information required for experiment is ready """
        if self._recommender is None:
            raise ValueError("recommender is not defined yet! user .load_recommender(cls) method to define.")
        if self._evaluator is None:
            raise ValueError("evaluator is not defined yet! use .load_evalutor(cls) method to define.")
        if self._clicker is None:
            raise ValueError("clicker is not defined yet! user .load_clicker(cls) method to define.")

        if self._init_user_connections is None:
            raise ValueError("now_user_connections is not defined!")
        if self._ref_user_connections is None:
            raise ValueError("ref_user_connections is not defined!")
        pass

    def _update_one_step(self):
        """ experiment advance by one iteration """
        max_iter = self._set_info["max_iter"]

        if self._is_directed:
            old_graph, new_graph = DiGraph(), DiGraph()
        else:
            old_graph, new_graph = Graph(), Graph()

        old_graph.add_edges_from(self._recommender._user_connections)
        old_tot_conns = len(old_graph.edges())

        new_connections = []
        tot_suggestions = 0
        tot_rejections  = 0
        if self._iteration < max_iter:
            start_time = datetime.now()

            uniq_user_ids = self._recommender._user_ids

            for ii, user_id in enumerate(uniq_user_ids):
                # retrieve recommended users
                # if user_id in self._rejected_user_dict:
                #    block_list = self._rejected_user_dict[user_id]
                # else:
                #    block_list = []

                suggestions = self._recommender.gen_suggestion(user_id)
                accepted, rejected = self._clicker.click(user_id, suggestions)

                tot_suggestions += len(suggestions)
                tot_rejections += len(rejected)

                # append new connections
                if len(accepted) > 0:
                    pairs = [[user_id, new_friend] for new_friend in accepted]
                    if len(new_connections) == 0:
                        new_connections = pairs
                    else:
                        new_connections.extend(pairs)

                # track rejected users
                # for some clicker simulator which, yeild rejected of empty
                # all the time to allow re-suggestions on recommended users
                if len(rejected) > 0:
                    if user_id in self._rejected_user_dict:
                        self._rejected_user_dict[user_id].extend(rejected)
                    else:
                        self._rejected_user_dict[user_id] = rejected

            # consolidate new connections
            new_connections = array(new_connections)

            # tracking experiment progress
            self._iteration += 1

            if new_connections.shape[0] > 0:
                # update simulator's connection data
                # self.load_init_user_connections(updated_user_connections)
                self._recommender.add_new_connections(new_connections)
                self._recommender.update()
                self._no_growth_counter = 0

            new_graph.add_edges_from(self._recommender._user_connections)
            new_tot_conns = len(new_graph.edges())
            new_added_conns = new_tot_conns - old_tot_conns

            duration = datetime.now() - start_time
            total_cost = duration.total_seconds()

            # collect evaluation scores
            self._evaluator.load_eval_user_connections(self._recommender._user_connections)
            eval_score = self._evaluator.get_score()

            cand_size = 0
            recommender_memory = copy.deepcopy(self._recommender._ordered_cand_dict)
            for k in recommender_memory.keys():
                cand_size += len(recommender_memory[k])

            # collect information
            exp_record = {"iteration": self._iteration,
                          "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                          "time_cost_seconds": total_cost,
                          "num_new_connections_size": new_added_conns,
                          "now_user_connections_size": len(self._recommender._user_connections),
                          "ref_user_connections_size": len(self._evaluator._ref_user_connections),
                          "tot_suggestions": tot_suggestions,
                          "tot_rejections": tot_rejections,
                          "tot_remain_candidates": cand_size
                          }
            exp_record.update(eval_score)

            # mark advacned of experiment
            self._recommender.update_iteration()

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

        # export experiment information
        fname = self.name + start_time.strftime("_%Y%m%d_%H%M%S.csv")
        outfile = join(self._outpath, fname)

        exp_records = []
        with tqdm(total=max_iter) as pbar:
            for ii in range(max_iter):
                #try:
                start_time = datetime.now()

                record = self._update_one_step()
                # collect time per iteration
                time_cost = (datetime.now() - start_time).total_seconds()
                record["time_cost_min"] = round(time_cost / 60, 1)
                exp_records.append(record)

                # write out test results
                if os.path.isfile(outfile):
                    # with open(outfile, mode='wr', encoding='utf-8') as f:
                    DataFrame([record]).to_csv(outfile, mode='a', header=False, index=False, sep=",")
                else:
                    DataFrame([record]).to_csv(outfile, header=True, index=False, sep=",")

                if self._no_growth_counter >= self._no_growth_max:
                    warnings.warn("experiment stops after reaching max number of no-growth iteration!")
                    break
                #except:
                #    msg = "".join(["Error Happend, experiment hault earlier than design!"])
                #    warnings.warn(msg)
                #    break
                pbar.update()

    def sys_reset(self):
        """ reset recommender's initial conenctions information for
        repeating experiment
        """
        self._recommender.load_user_connections(self._init_user_connections)
        self._recommender._iter_counter = 0
        self._iteration = 0
        self._no_growth_counter = 0

