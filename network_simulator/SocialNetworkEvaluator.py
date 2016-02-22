""" evaluator for comparing two social networks
Author: Yi Zhang <beingzy@gmail.com>
Date: 2016/02/20
"""


class EvaluatorMixin(object):

    def __init__(self):
        self._ref_user_connections = None
        self._eval_uesr_connections = None

    def load_ref_user_connections(self, user_connections):
        self._ref_user_connections = user_connections

    def load_eval_user_connections(self, user_connections):
        self.load_eval_user_connections = user_connections


class SocialNetworkEvaluator(EvaluatorMixin):

    def _get_similiarity(self, ref_user_connections, eval_user_connections):
        return 0

    def _get_dissimilarity(self, ref_user_connections, eval_user_connections):
        return 0

    def get_score(self, ref_user_connections=None, eval_user_connections=None):

        if ref_user_connections is None:
            ref_user_connections = self._ref_user_connections

        if eval_user_connections is None:
            eval_user_connections = self.load_eval_user_connections

        score_sim = self._get_similiarity(ref_user_connections, eval_user_connections)
        score_dissim = self._get_dissimilarity(ref_user_connections, eval_user_connections)

        res = {'score_sim': score_sim, 'score_dissim': score_dissim}
        return res
