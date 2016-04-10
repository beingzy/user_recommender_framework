""" evaluator for comparing two social networks
Author: Yi Zhang <beingzy@gmail.com>
Date: 2016/02/20
"""
import numpy as np
import networkx as nx
from networkx import Graph, DiGraph


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
            user_connections = _normalize_connections(user_connections, self._is_directed)
        self._ref_user_connections = user_connections

    def load_eval_user_connections(self, user_connections):
        """"""
        if isinstance(user_connections, np.ndarray):
            user_connections = user_connections.tolist()
        if isinstance(user_connections, list):
            # make copy of immutable list
            user_connections = user_connections[:]
        if not self._is_directed:
            user_connections = _normalize_connections(user_connections, self._is_directed)
        self._eval_user_connections = user_connections

    def get_directed_status(self):
        return self._is_directed


class SocialNetworkEvaluator(EvaluatorMixin):
    """ social network comparison tool

    Parameters:
    ----------
    is_directed: boolean, defaulted = False
        define how to treat network as directed or  undirected
    """

    def get_similiarity(self, ref_user_connections, eval_user_connections, is_directed=False):
        # convert array to list
        if isinstance(ref_user_connections, np.ndarray):
            ref_user_connections = ref_user_connections.tolist()

        if isinstance(eval_user_connections, np.ndarray):
            eval_user_connections = eval_user_connections.tolist()

        cdr_score = common_edge_ratio(ref_user_connections, eval_user_connections, is_directed)
        eigenvec_score = eigenvector_similarity(ref_user_connections, eval_user_connections, is_directed)
        return cdr_score, eigenvec_score

    def get_dissimilarity(self, ref_user_connections, eval_user_connections):
        return None

    def get_score(self, ref_user_connections=None, eval_user_connections=None):

        if ref_user_connections is None:
            ref_user_connections = self._ref_user_connections

        if eval_user_connections is None:
            eval_user_connections = self._eval_user_connections

        cdr_score, eigenvec_score = self.get_similiarity(ref_user_connections, eval_user_connections)
        score_dissim = self.get_dissimilarity(ref_user_connections, eval_user_connections)

        res = {'common_edge_ratio': cdr_score,
               'eigenvector_similarity': eigenvec_score,
               'score_dissim': score_dissim}
        return res


def common_edge_ratio(ref_user_connections, eval_user_connections, is_directed=False):
    """ caulcalate the fraction of common edges fraction out of union of two graphs

    Parameters:
    ==========
    ref_user_connections: a list of edges
    eval_user_connections: a list of edges
    is_directed: boolean,
        False (default): edges forms an undirected graph
        True: edges forms a directed graph
    """
    tot_common = sum([1 for item in eval_user_connections if item in ref_user_connections])
    union_size = len(ref_user_connections) + len(eval_user_connections) - tot_common
    return tot_common / union_size


def eigenvector_similarity(ref_user_connections, eval_user_connections, is_directed=False):
    """ calculate transformed (s = 1/(1-d) )eigevector similiarity

    technique mentoioned in paper: https://www.cs.cmu.edu/~jingx/docs/DBreport.pdf
    code resource: http://goo.gl/XauaWB

    Parameters:
    ==========
    ref_user_connections: a list of edges
    eval_user_connections: a list of edges
    is_directed: boolean,
        False (default): edges forms an undirected graph
        True: edges forms a directed graph
    """
    if is_directed:
        ref_graph, eval_graph = DiGraph(), DiGraph()
        ref_graph.add_edges_from(ref_user_connections)
        eval_graph.add_edges_from(eval_user_connections)
    else:
        ref_graph, eval_graph = Graph(), Graph()
        ref_graph.add_edges_from(ref_user_connections)
        eval_graph.add_edges_from(eval_user_connections)

    def select_k(spectrum, minimum_energy=0.9):
        running_total = 0.0
        total = sum(spectrum)
        if total == 0.0:
            return len(spectrum)
        for i in range(len(spectrum)):
            running_total += spectrum[i]
            if running_total / total >= minimum_energy:
                return i + 1
        return len(spectrum)

    ref_laplacian = nx.spectrum.laplacian_spectrum(ref_graph)
    eval_laplacian = nx.spectrum.laplacian_spectrum(eval_graph)

    k1 = select_k(ref_laplacian)
    k2 = select_k(eval_laplacian)
    k = min(k1, k2)
    score = sum((ref_laplacian[:k] - eval_laplacian[:k])**2)

    # original score is unbounded,
    # returns 0 for two identical graphs,
    # a larger value indicates a greater difference
    # convert the score from disance-style to similarity-style
    return 1.0 / (1.0 + score)


def _normalize_connections(connections, is_directed=False):
    """ convert numpy.array of connections to a list of set items to represent undirected connections."""
    if not is_directed:
         connections = [set(pair) for pair in connections]

    uniq_connections = []
    for item in connections:
        if not item in uniq_connections:
            uniq_connections.append(item)

    return uniq_connections