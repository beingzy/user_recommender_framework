""" unit-test file for social network evaluator
Author: Yi Zhang <beingzy@gamil.com>
Date: 2016/02/22
"""
import unittest
from numpy import array
from network_simulator.SocialNetworkEvaluator import SocialNetworkEvaluator


class TestSocialNetworkEvaluator(unittest.TestCase):

    def setUp(self):
        ref_user_connections = array([])
        eval_user_connections = array([])

        self._evaluator = SocialNetworkEvaluator()
        self._evaluator.load_ref_user_connections(ref_user_connections)
        self._evaluator.load_eval_user_connections(eval_user_connections)

    def test_init_SocialNetworkEvaluator(self):
        eval_scores = self._evaluator.get_score()
        is_match = eval_scores["sim_score"] == 0 and eval_scores["dissim_score"] == 0
        self.assertTrue(is_match)
