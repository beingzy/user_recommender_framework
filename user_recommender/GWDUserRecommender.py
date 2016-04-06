""" User Recommender with Groupwise Distance Learning

Author: Yi Zhang <beingzy@gmail.com>
Date: 2016/04/05
"""
from user_recommender import UserRecommenderMixin
from distance_metrics import GeneralDistanceWrapper
from user_recommender.PairwiseDistMatrix import PairwiseDistMatrix
from groupwise_distance_learning.groupwise_distance_learner import GroupwiseDistLearner

def _consolidate_learned_info(gwd_learner, buffer_min_size):
    """ process buffer group's users either to stay independent
        or merge into largest group.
        It returns a consolidated set of fit_weights and fit_group

        Parameters:
        ===========
        * gwd_learner: GroupWiseDistLearner instance
        * buffer_min_size: numeric
    """
    if not isinstance(gwd_learner, GroupwiseDistLearner):
        msg = "".join(["invalid input, gwd_learner must be",
                       "an instance of GroupwiseDistLearner!"])
        raise ValueError(msg)

    # extract learned information
    fit_weights = gwd_learner.get_groupwise_weights()
    fit_groups, buffer_group = gwd_learner.get_user_cluster()

    # process buffer group
    group_ids = list(fit_groups.keys())
    buffer_group_size = len(buffer_group)
    if buffer_group_size > 0:

        if buffer_group_size < buffer_min_size:
            group_sizes = [len(fit_groups[gid]) for gid in group_ids]
            # select first one among ties
            largest_group_id = [gid for gid, gsize in zip(group_ids, group_sizes) \
                                if gsize == max(group_sizes)][0]
            fit_groups[largest_group_id].extend(buffer_group)
        else:
            buffer_group_id = max(group_ids) + 1
            group_ids.append(buffer_group_id)
            # add buffer group weights
            n_feats = len(fit_weights[group_ids[0]])
            fit_weights[buffer_group_id] = [1.0] * n_feats
            fit_groups[buffer_group_id] = buffer_group

    return fit_weights, fit_groups


class GWDUserRecommender(UserRecommenderMixin):

    def __init__(self, user_ids, user_profiles, user_connections, **kwargs):
        super().__init__()
        # load user-related information
        self.load_user_ids(user_ids)
        self.load_user_profiles(user_profiles)
        self.load_user_connections(user_connections)

        # equip with GWD leanner
        self.gwd_learner = GroupwiseDistLearner(**kwargs)
        # initiate learning
        self._triger_groupwise_learning()
        #
        _gw_dist_matrics = self.gwd_learner.get_groupwise_weights()
        _learned_group, _buffer_group = self.gwd_learner.get_user_cluster()


    def _triger_groupwise_learning(self):
        self.gwd_learner.fit(self._user_ids,
                             self._user_profiles,
                             self._user_connections)


    def _update_dist_func(self):
        pass

    def update(self, **kwargs):
        """ update social network
        """
        pass

    def add_new_connections(self, new_user_connections):
        pass

    def get_connected_users(self, user_ids):
        pass

    def gen_suggestion(self, user_id):
        pass