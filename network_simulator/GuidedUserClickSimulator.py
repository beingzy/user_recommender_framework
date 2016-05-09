""" objective introduction

Author: Yi Zhang <beingzy@gmail.com>
Date: 2016/05/09
"""
from .UserClickSimulator import UserClickSimulatorMixin


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


class GuidedUserClickSimulator(UserClickSimulatorMixin):

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
        rejected = []
        for ii, uid in enumerate(rec_list):
            if uid in known_user_conns:
                accepted.append(uid)
            else:
                rejected.append(uid)

        return accepted, rejected
