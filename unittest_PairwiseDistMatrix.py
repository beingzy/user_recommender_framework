""" unit-test for pairwise distance matrix
"""
import unittest
from os import getcwd
from os.path import join
from pandas import read_csv
from PairwiseDistMatrix import PairwiseDistMatrix

class TestPairwiseDistMatrix(unittest.TestCase):

    def setUp(self):
        """load test data"""
        _ROOT_DIR = getcwd()
        _DATA_DIR = join(_ROOT_DIR, "data")

        # set up data
        user_profile_df = read_csv( join(_DATA_DIR, 'user_profile.csv'), header=0)
        user_ids = user_profile_df['id'].tolist()
        user_profiles = user_profile_df.drop(['id'], axis=1).as_matrix()

        # create DistMatrix Distance
        self.dist_matrix = PairwiseDistMatrix(user_ids, user_profiles)
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

    def test_default_calculation(self):
        """ test correctness of calculated distance """
        # access distance involving user_id: a
        user_list, dist_list = self.dist_matrix.list_all_dist(user_id='a')
        b_idx = user_list.index('b')
        cal_dist_ab = dist_list[b_idx]
        true_dist_ab = 1.1148012717248872
        self.assertEqual( cal_dist_ab, true_dist_ab )


if __name__ == "__main__":
    unittest.main()