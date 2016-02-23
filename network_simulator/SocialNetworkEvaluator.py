""" evaluator for comparing two social networks
Author: Yi Zhang <beingzy@gmail.com>
Date: 2016/02/20
"""
import numpy as np


def normalize_connections(connections):
    """ convert numpy.array of connections to a list of set
    items to represent undirected connections.
    """
    connections = [set(pair) for pair in connections]
    uniq_connections = []
    for item in connections:
        if not item in uniq_connections:
            uniq_connections.append(item)
    return uniq_connections


class EvaluatorMixin(object):

    def __init__(self, is_directed=False):
        self._ref_user_connections = None
        self._eval_uesr_connections = None
        self._is_directed = is_directed

    def load_ref_user_connections(self, user_connections):
        if isinstance(user_connections, np.ndarray):
            user_connections = user_connections.tolist()
        if isinstance(user_connections, list):
            # make copy of immutable list
            user_connections = user_connections[:]
        if not self._is_directed:
            user_connections = normalize_connections(user_connections)
        self._ref_user_connections = user_connections

    def load_eval_user_connections(self, user_connections):
        """"""
        if isinstance(user_connections, np.ndarray):
            user_connections = user_connections.tolist()
        if isinstance(user_connections, list):
            # make copy of immutable list
            user_connections = user_connections[:]
        if not self._is_directed:
            user_connections = normalize_connections(user_connections)
        self._eval_user_connections = user_connections

    def get_directed_status(self):
        return self._is_directed


class SocialNetworkEvaluator(EvaluatorMixin):
    def get_similiarity(self, ref_user_connections, eval_user_connections):
        # convert array to list
        if isinstance(ref_user_connections, np.ndarray):
            ref_user_connections = ref_user_connections.tolist()
        if isinstance(eval_user_connections, np.ndarray):
            eval_user_connections = eval_user_connections.tolist()

        tot_common = sum([1 for item in eval_user_connections if item in ref_user_connections])
        union_size = len(ref_user_connections) + len(eval_user_connections) - tot_common
        return tot_common / union_size

    def get_dissimilarity(self, ref_user_connections, eval_user_connections):
        return None

    def get_score(self, ref_user_connections=None, eval_user_connections=None):

        if ref_user_connections is None:
            ref_user_connections = self._ref_user_connections

        if eval_user_connections is None:
            eval_user_connections = self._eval_user_connections

        score_sim = self.get_similiarity(ref_user_connections, eval_user_connections)
        score_dissim = self.get_dissimilarity(ref_user_connections, eval_user_connections)

        res = {'score_sim': score_sim, 'score_dissim': score_dissim}
        return res
