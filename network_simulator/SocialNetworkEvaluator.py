""" evaluator for comparing two social networks
Author: Yi Zhang <beingzy@gmail.com>
Date: 2016/02/20
"""


class SocialNetworkEvaluator(object):
    def __init__(self):
        pass

    def load_ref_user_connections(self, user_connections):
        self._ref_user_connections = user_connections

    def load_eval_user_connections(self, user_connections):
        self.load_eval_user_connections = user_connections

    def _get_similiarity(self, ref_user_connections, eval_user_connections):
        return 0

    def _get_dissimilarity(self, ref_user_connections, eval_user_connections):
        return 0

    def get_score(self, ref_user_connections=None, eval_user_connections=None):

        if ref_user_connections is None:
            ref_user_connections = self._ref_user_connections
        if eval_user_connections is None:
            eval_user_connections = self.load_eval_user_connections

        sim_score = self._get_similiarity(ref_user_connections, eval_user_connections)
        dissim_score = self._get_dissimilarity(ref_user_connections, eval_user_connections)

        res = {'sim_score': sim_score, 'dissim_score': dissim_score}
        return res
