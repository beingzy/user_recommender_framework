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


class UserClickSimulator(object):
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
