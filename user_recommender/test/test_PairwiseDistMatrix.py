""" unit-test for pairwise distance matrix
Author: Yi Zhang <beingzy@gmail.com>
Date: 2016/02/20
"""
import unittest
from os import getcwd
# load helpfer function
from user_recommender.test.helper_func import load_test_data
# import test package
from user_recommender import PairwiseDistMatrix


class TestPairwiseDistMatrix(unittest.TestCase):

    def setUp(self):
        """load test data"""
        _ROOT_DIR = getcwd()
        # set up data
        user_ids, user_profiles, _ = load_test_data( _ROOT_DIR)
        # create DistMatrix Distance
        self.dist_matrix = PairwiseDistMatrix(user_ids, user_profiles)
        # calculate distance metrix
        self.dist_matrix.update_distance_matrix()

    def test_attributes(self):
        """ test if PairwiseDistMatrix has all the attributes """
        actual_attr = list( self.dist_matrix.__dict__.keys() )
        defined_attr = ['_dist_user_pairs', '_update_status', '_dist_func', '_dist_matrix', \
                        '_user_profiles', '_user_ids']
        tot_impl_attr = sum([ 1 for attr in actual_attr if attr in defined_attr ])
        self.assertEqual( tot_impl_attr, len(defined_attr) )

    def test_update_state(self):
        """ test _update_status, it is False after update_distance_matrix() """
        self.assertEqual(self.dist_matrix._update_status, False)

    def test_access_distance(self):
        """ test return of .list_all_dist """
        user_list, _ = self.dist_matrix.list_all_dist(user_id='a')
        defined_user_list = ['b', 'c', 'd', 'e']
        tot_matches = sum([ 1 for uid in user_list if uid in defined_user_list ])
        self.assertEqual( tot_matches, len(defined_user_list) )

    def test_default_calculation_sam_cat(self):
        """ test correctness of calculated distance """
        # access distance involving user_id: a
        user_list, dist_list = self.dist_matrix.list_all_dist(user_id='a')
        b_idx = user_list.index('b')
        cal_dist = round(dist_list[b_idx], 3)
        true_dist = round(1.1148012717248872, 3)
        self.assertEqual( cal_dist, true_dist )

    def test_default_calculation_diff_cat(self):
        """ test correctness of calculated distance """
        # access distance involving user_id: a
        user_list, dist_list = self.dist_matrix.list_all_dist(user_id='a')
        idx = user_list.index('c')
        cal_dist = round(dist_list[idx], 3)
        true_dist = round(1.5136858414837358, 3)
        self.assertEqual( cal_dist, true_dist )


if __name__ == "__main__":
    unittest.main()