""" unit-test for Nearest Neigbour User Recommendation System
Author: Yi Zhang <beingzy@gmail.com>
Date: 2016/02/20
"""
import unittest
from os import getcwd
# load helpfer function
from test.helper_func import load_test_data
# import test package
from user_recommender.NNUserRecommender import NNUserRecommender

class TestNNUserRecommender(unittest.TestCase):

    def setUp(self):
        """load test data"""
        _ROOT_DIR = getcwd()
        # set up data
        user_ids, user_profiles, user_connections = load_test_data( _ROOT_DIR)
        # initiate NNRecSys
        self.nnrec_sys = NNUserRecommender(user_ids, user_profiles, user_connections)
        self.nnrec_sys.set_recommendation_size(2)
        self.nnrec_sys.set_max

    def test_list_connceted_users(self):
        returned_user_ids = self.nnrec_sys.get_connected_users(user_id='a')
        defined_user_ids = ['b', 'd', 'e']
        is_match = set(returned_user_ids) == set(defined_user_ids)
        self.assertTrue(is_match)

    def test_gen_suggestions(self):
        returned_rec = self.nnrec_sys.gen_suggestion('a')
        possible_rec = ['c']
        is_match = returned_rec == possible_rec
        self.assertTrue(is_match)

    def test_add_new_connections(self):
        self.nnrec_sys.add_new_connections([['a', 'c']])
        returned_user_ids = self.nnrec_sys.get_connected_users(user_id='a')
        defined_user_ids = ['b', 'c', 'd', 'e']
        is_match = set(returned_user_ids) == set(defined_user_ids)
        self.assertTrue(is_match)

    def test_gen_suggestions_after_new_connections(self):
        self.nnrec_sys.add_new_connections([['a', 'c']])
        returned_rec = self.nnrec_sys.gen_suggestion('a')
        possible_rec = []
        is_match = returned_rec == possible_rec
        self.assertTrue(is_match)


if __name__ == "__main__":
    unittest.main()