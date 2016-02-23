""" Unit-test file for UserRecSysExpSimulator
Author: Yi Zhang
Date: 2016/02/20
"""
import unittest
from itertools import combinations
from os import getcwd
# load helpfer function
from user_recommender.test.helper_func import load_test_data
# load depdendent class
from user_recommender.UserRecommenderMixin import UserRecommenderMixin
from user_recommender.NNUserRecommender import NNUserRecommender
# load test class
from network_simulator.UserRecSysExpSimulator import UserRecSysExpSimulator
from network_simulator.SocialNetworkEvaluator import SocialNetworkEvaluator
from network_simulator.UserClickSimulator import UserClickSimulator


class TestUserRecSysExpSimiulator(unittest.TestCase):

    def setUp(self):
        _ROOT_DIR = getcwd()
        # set up data
        user_ids, user_profiles, user_connections = load_test_data(_ROOT_DIR)
        # initiate NNRecSys
        self._simulator = UserRecSysExpSimulator(name="test simulator")
        self._simulator.load_init_data(user_ids=user_ids,
                                       user_profiles=user_profiles,
                                       user_connections=user_connections)
        # set the referrence user connection
        complete_connecions = [list(p) for p in combinations(user_ids, 2)]
        self._simulator.load_referrence_data(complete_connecions)

        # experiment setting
        self._simulator.set_recommendation_size(5)
        self._simulator.set_max_iterations(10)

    def test_all_attributes(self):
        self.assertEqual(self._simulator.name, "test simulator")

    def test_load_and_init_UserRecommenderMixin(self):
        self._simulator.load_recommender(UserRecommenderMixin)
        recommender_rec_size = self._simulator._recommender._size
        self.assertEqual(recommender_rec_size, 5)

    def test_load_and_init_NNUserRecommender(self):
        recommender_cls = NNUserRecommender
        self._simulator.load_recommender(recommender_cls)
        recommender_rec_size = self._simulator._recommender._size
        self.assertEqual(recommender_rec_size, 5)

    def test_run_experiment(self):
        recommender_cls = NNUserRecommender
        dummy_clicker_cls = UserClickSimulator
        null_evaluator_cls = SocialNetworkEvaluator

        self._simulator.load_recommender(recommender_cls)
        self._simulator.load_clicker(dummy_clicker_cls)
        self._simulator.load_evaluator(null_evaluator_cls)

        self._simulator.set_max_iterations(10)
        self._simulator.run()

        is_done = True
        self.assertTrue(is_done)


if __name__ == "__main__":
    unittest.main()
