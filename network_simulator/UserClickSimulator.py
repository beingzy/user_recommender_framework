""" user click simulator to simulate the user operation which is reactive to viewing recommendations
Author: Yi Zhang <beingzy@gmail.com>
Date: 2016/02/20
"""
from numpy.random import choice


def zipf_pdf(k, n, s=1):
    """ return the probability of nth rank

        Parameters:
        ----------
        k: {int} kth rank
        n: {int} total number of elemnets
        s: {float} distribution parameter
    """
    num = 1.0 / (k ** s)
    den = sum([1.0 / ((ii + 1) ** s) for ii in range(n)])
    return num / den


class UserClickSimulatorMixin(object):

    def click(self, user_id, rec_list):
        return [], []


class UserClickSimulator(UserClickSimulatorMixin):

    def click(self, target_user_id, rec_list):
        # simulation process: generate recommendation list for user
        # user clicks list of recommended users how many times the user would click
        # this experiment style allows re-suggestions on recommended users
        click_size = choice([0, 1], p=[0.2, 0.8])

        accepted, rejected = [], []
        if click_size > 0 and len(rec_list) > 0:
            # this method is contrained by _prob_distributor's maximum
            # allowed
            n = len(rec_list)
            click_probs = [zipf_pdf(ii + 1, n, s=1) for ii in range(n)]
            accepted = choice(rec_list, click_size, False, click_probs)
            accepted = list(accepted)
        else:
            accepted = []
            rejected = []

        rejected = [uid for uid in rec_list if not (uid in accepted)]
        return accepted, rejected


class GuidedUserClickSimulator(UserClickSimulatorMixin):

    def __init__(self, reference_user_connections):
        """
        Parameters:
        ==========
        reference_user_connections: {matrix-like} (n, 2)
            pair of users to represent true status of social network
        """
        reference_user_connections = _convert_pair_dictionary(reference_user_connections)
        self._ref_user_connections = reference_user_connections

    def click(self, target_user_id, rec_list):
        known_user_conns = self._ref_user_connections[target_user_id]
        accepted, rejected = [], []
        for ii, uid in enumerate(rec_list):
            if uid in known_user_conns:
                accepted.append(uid)
            else:
                rejected.append(uid)

        return accepted, rejected


def _convert_pair_dictionary(user_connections, is_directed=False):
    """ translate user pairs into user-wise dictionary
    """
    user_conn_dict = {}
    for ii, (uid_a, uid_b) in enumerate(user_connections):

        if uid_a in user_conn_dict:
            user_conn_dict[uid_a].append(uid_b)
        else:
            user_conn_dict[uid_a] = [uid_b]

        if not is_directed:
            if uid_b in user_conn_dict:
                user_conn_dict[uid_b].append(uid_a)
            else:
                user_conn_dict[uid_b] = [uid_a]

    return user_conn_dict

