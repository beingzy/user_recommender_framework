""" objective introduction

Author: Yi Zhang <beingzy@gmail.com>
Date: 2016/05/09
"""
import unittest
from ..UserClickSimulator import GuidedUserClickSimulator


class TestGuidedUserClickSimulator(unittest.TestCase):

    def __init__(self):
        self._user_connections = [[1, 2], [1, 3],
                                  [2, 3], [2, 5],
                                  [3, 5], [4, 5],
                                  [4, 6]]

    def TestRec(self):
        user_conns = self._user_connections
        guided_clicker = GuidedUserClickSimulator(user_conns)
        accepted, recommended = guided_clicker(1, [2, 3, 4, 5, 6])
        self.assertEqual(accepted, [2, 3])
        self.assertEqual(recommended, [4, 5, 6])
