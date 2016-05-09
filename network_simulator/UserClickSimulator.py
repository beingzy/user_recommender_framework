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


def convert_pair_dictionary(user_connections):
    """ translate user pairs into user-wise dictionary
    """
    user_conn_dict = {}
    for ii, (uid_a, uid_b) in enumerate(user_connections):

        if uid_a in user_conn_dict:
            user_conn_dict[uid_a].append(uid_b)
        else:
            user_conn_dict[uid_a] = [uid_b]

        if uid_b in user_conn_dict:
            user_conn_dict[uid_b].append(uid_a)
        else:
            user_conn_dict[uid_b] = [uid_a]

    return user_conn_dict


class UserClickSimulatorMixin(object):

    def click(self, rec_list):
        return rec_list


class UserClickSimulator(UserClickSimulatorMixin):

    def click(self, rec_list):
        # simulation process: generate recommendation list for user
        # user clicks list of recommended users how many times the user would click
        click_size = choice([0, 1], p=[0.2, 0.8])

        if click_size > 0 and len(rec_list) > 0:
            # this method is contrained by _prob_distributor's maximum
            # allowed
            n = len(rec_list)
            click_probs = [zipf_pdf(ii + 1, n, s=1) for ii in range(n)]
            confirmed = choice(rec_list, click_size, False, click_probs)
            confirmed = list(confirmed)
        else:
            confirmed = []

        return confirmed


class GuidedUserClickSimulatior(UserClickSimulatorMixin):

    def __init__(self, reference_user_connections):
        """
        Parameters:
        ==========
        reference_user_connections: {matrix-like} (n, 2)
            pair of users to represent true status of social network
        """
        reference_user_connections = convert_pair_dictionary(reference_user_connections)
        self._ref_user_connections = reference_user_connections

    def click(self, target_user_id, rec_list):
        known_user_conns = self._ref_user_connections[target_user_id]
        accepted = []
        recommended = []
        for ii, uid in enumerate(rec_list):
            if uid in known_user_conns:
                accepted.append(uid)
            else:
                recommended.append(uid)

        return accepted, recommended

