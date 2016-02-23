""" unit-test file for social network evaluator
Author: Yi Zhang <beingzy@gamil.com>
Date: 2016/02/22
"""
import unittest
from numpy import array
from network_simulator.SocialNetworkEvaluator import SocialNetworkEvaluator


class TestSocialNetworkEvaluator(unittest.TestCase):

    def setUp(self):
        ref_user_connections = array([[1, 2], [2, 3], [3, 2]])
        eval_user_connections = array([[3, 2], [3, 1], [3, 4]])

        self._evaluator = SocialNetworkEvaluator(is_directed=False)
        self._evaluator.load_ref_user_connections(ref_user_connections)
        self._evaluator.load_eval_user_connections(eval_user_connections)

    def test_score_sim(self):
        scores = self._evaluator.get_score()
        self.assertEqual(scores["score_sim"], 0.25)

    def test_score_sim_same(self):
        """ test same ref_user_connections and eval_user_conenctions """
        eval_user_connections = array([[1, 2], [2, 3]])
        self._evaluator.load_eval_user_connections(eval_user_connections)
        scores = self._evaluator.get_score()
        self.assertEqual(scores["score_sim"], 1)

    def test_directed_score_sim(self):
        ref_user_connections = array([[1, 2], [2, 3], [3, 2]])
        eval_user_connections = array([[3, 2], [3, 1], [3, 4]])

        directed_evaluator = SocialNetworkEvaluator(is_directed=True)
        directed_evaluator.load_ref_user_connections(ref_user_connections)
        directed_evaluator.load_eval_user_connections(eval_user_connections)
        scores = directed_evaluator.get_score()
        self.assertEqual(scores["score_sim"], 0.2)

    def test_score_disim(self):
        scores = self._evaluator.get_score()
        is_match = scores["score_dissim"] is None
        self.assertTrue(is_match)


if __name__ == "__main__":
    unittest.main()
