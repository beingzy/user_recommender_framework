""" User Recommender with Groupwise Distance Learning

Author: Yi Zhang <beingzy@gmail.com>
Date: 2016/04/05
"""
from numpy import array, vstack
from pandas import DataFrame
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
        """ instance initilization function

        Parameters:
        ===========
        * user_ids: <list> a list of unique user ids
        * user_profiles: <matrix-like> a numpy.array of user profile
        * target_uer_ids: <list> a list of user whose distance in relation to other users
            will be  calculated only.
        """

        super().__init__()
        if isinstance(user_profiles, DataFrame):
            user_profiles = user_profiles.as_matrix()

        if len(user_ids) != user_profiles.shape[0]:
            nobs = len(user_ids)
            nrow = user_profiles.shape[0]
            raise ValueError("user_ids (nobs: %d) does not match user_profiles (nrow: %d)".format(nobs, nrow))

        # collect parameters from kwargs, this argument
        # detemine if buffer group should become an independent group
        # with even weights for recommendation
        try:
            self._buffer_min_size = kwargs["buffer_min_size"]
        except:
            self._buffer_min_size = None # load function default value

        # load user-related information
        self.load_user_ids(user_ids)
        self.load_user_profiles(user_profiles)
        self.load_user_connections(user_connections)

        # initiate containers of PairwiseDistMatrix
        # which store ciritical information for recommendation
        # generation
        self._pdm_container = {}

        # Groupwise Distance Learner output container
        self._fit_weights = {}
        self._fit_groups = {}
        self._all_group_ids = []

        # equip with GWD leanner
        self.gwd_learner = GroupwiseDistLearner(**kwargs)
        # initiate learning
        self._triger_groupwise_learning()
        # marker for information updated
        self._is_updated = False

    def _return_user_group(self, user_id):
        """ return group key of a given user
        """
        for gid in self._all_group_ids:
            if user_id in self._fit_groups[gid]:
                return gid

    def _triger_groupwise_learning(self):
        self.gwd_learner.fit(self._user_ids,
                             self._user_profiles,
                             self._user_connections)

        fit_weights, fit_groups = _consolidate_learned_info(self.gwd_learner,
                                                            self._buffer_min_size)

        # process update pairwise distance matrix
        group_ids = fit_groups.keys()
        pdm_container = {}

        for ii, group_id in enumerate(group_ids):
            # load weighted distance
            target_weights = fit_weights[group_id]
            temp_dist_wrapper = GeneralDistanceWrapper()
            temp_dist_wrapper.fit(self._user_profiles)
            temp_dist_wrapper.load_weights(target_weights)
            target_dist_func = temp_dist_wrapper.dist_euclidean

            # load distance matrix container
            target_user_ids = fit_groups[group_id]
            pdm = PairwiseDistMatrix(self._user_ids, self._user_profiles,
                                     target_user_ids=target_user_ids)
            # load weighted distance calculation and
            pdm.set_dist_func(target_dist_func)
            pdm.update_distance_matrix()

            pdm_container[group_id] = pdm

        # update attribute
        self._pdm_container = pdm_container
        self._fit_weights = fit_weights
        self._fit_groups = fit_groups
        self._all_group_ids = list(fit_groups.keys())

    def set_recommendation_size(self, size=5):
        self._size = size

    def update(self, **kwargs):
        """ update social network
        """
        if "user_ids" in kwargs.keys():
            self.load_user_ids(kwargs["user_ids"])
            self._is_updated = True

        if "user_profiles" in kwargs.keys():
            self.load_user_profiles(kwargs["user_profiles"])
            self._is_updated = True

        if "user_connections" in kwargs.keys():
            self.load_user_connections(kwargs["user_connections"])
            self._is_updated = True

        # MUST UPDATE EITHER BOTH OF user_ids, user_profiles
        # OR NONE OF THEM !!!
        # ADD VALIDATION FUNCTION IS NEEDED
        if self._is_updated:
            # either
            self._triger_groupwise_learning()
            self._is_updated = False

    def add_new_connections(self, new_user_connections):
        if isinstance(new_user_connections, list):
            if isinstance(new_user_connections[0], list):
                new_user_connections = array(new_user_connections)
            else:
                raise ValueError("illegal new_user_connections data is provided !")
        self._user_connections = vstack((self._user_connections, new_user_connections))
        self._is_updated = True

    def get_connected_users(self, user_id):
        """ return a list of user who are connceted with the target user"""
        b_user_ids = [pp[1] for pp in self._user_connections if pp[0] == user_id]
        a_user_ids = [pp[0] for pp in self._user_connections if pp[1] == user_id]
        return list(set(b_user_ids + a_user_ids))

    def gen_suggestion(self, user_id):
        # get all condadaite user
        user_gid = self._return_user_group(user_id)
        cand_user_ids, cand_user_dist = self._pdm_container[user_gid].list_all_dist(user_id)
        con_user_ids = self.get_connected_users(user_id)

        # remove connected users from condidate list
        keep_idx = [ii for ii, cand_user_id in enumerate(cand_user_ids) if not cand_user_id in con_user_ids]
        cand_user_ids = [cand_user_ids[ii] for ii in keep_idx]
        cand_user_dist = [cand_user_dist[ii] for ii in keep_idx]

        # sort candidates by distance
        sorted_list = sorted(zip(cand_user_ids, cand_user_dist), key=lambda pp: pp[1])
        sorted_cand_uids = [uid for uid, _ in sorted_list]

        size = self._size
        if len(sorted_cand_uids) > size:
            return sorted_cand_uids[:size]
        else:
            return sorted_cand_uids