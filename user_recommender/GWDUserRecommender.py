""" User Recommender with Groupwise Distance Learning

Author: Yi Zhang <beingzy@gmail.com>
Date: 2016/04/05
"""
from numpy import array, vstack
from pandas import DataFrame
from . import UserRecommenderMixin
from .PairwiseDistMatrix import PairwiseDistMatrix
from ..groupwise_distance_learning.groupwise_distance_learner import GroupwiseDistLearner
from ..distance_metrics import GeneralDistanceWrapper


def _consolidate_learned_info(gwd_learner, buffer_min_size=None):
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

    if buffer_min_size is None:
        buffer_min_size = 20

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

    def __init__(self, user_ids, user_profiles, user_connections, only_init_learn=False,
                 update_iter_period=None, **kwargs):
        """ instance initilization function

        Parameters:
        ===========
        * user_ids: <list> a list of unique user ids
        * user_profiles: <matrix-like> a numpy.array of user profile
        * target_uer_ids: <list> a list of user whose distance in relation to other users
            will be  calculated only.
        * only_init_learn: boolean
            True: GroupwiseDistLearner only learn user_group and groupwise distance at one time
            False: GroupwiseDistLearner will learn every time
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

        # store attribute
        self._only_init_learn = only_init_learn

        # load user-related information
        self.load_user_ids(user_ids)
        self.load_user_profiles(user_profiles)
        self.load_user_connections(user_connections)

        # initiate containers of PairwiseDistMatrix
        # which store ciritical information for recommendation
        # generation
        self._pdm_container = {}
        # store orderred unconnected users
        # and the container will be reset per distance metrics update
        self._ordered_cand_dict = {}
        # store users had been ever recommended
        self._recommended_user_dict = {}

        # Groupwise Distance Learner output container
        self._fit_weights = {}
        self._fit_groups = {}
        self._all_group_ids = []

        # define update frequency
        self._update_iter_period = update_iter_period
        if not update_iter_period is None:
            # overwrite init_ if update_iter_period had been set
            self._only_init_learn = False

        # equip with GWD leanner
        self.gwd_learner = GroupwiseDistLearner(**kwargs)
        # initiate learning
        self._triger_groupwise_learning()
        # marker for information updated
        self._is_updated = False

        # load distance wrapper
        self._gd_wrapper = GeneralDistanceWrapper()
        self._gd_wrapper.fit(user_profiles)

    def _get_distance(self, a_user_id, b_user_id):
        """ calcualte two users's distance from user a's group weights
        """
        a_user_idx = [i for i, uid in enumerate(self._user_ids) if uid == a_user_id][0]
        b_user_idx = [i for i, uid in enumerate(self._user_ids) if uid == b_user_id][0]
        a_user_profile = self._user_profiles[a_user_idx, :]
        b_user_profile = self._user_profiles[b_user_idx, :]

        gid = self._return_user_group(a_user_id)
        weights = self._fit_weights[gid]
        self._gd_wrapper.load_weights(weights)
        return self._gd_wrapper.dist_euclidean(a_user_profile, b_user_profile)

    def _return_user_group(self, user_id):
        """ return group key of a given user
        """
        for gid in self._all_group_ids:
            if user_id in self._fit_groups[gid]:
                return gid

    def _triger_groupwise_learning(self):
        # initial the learning of embedded GDL algorithm
        if self._iter_counter == 0:
            # avoid repeated initial learning
            if len(self._fit_weights) > 0:
                fit_weights, fit_groups = _consolidate_learned_info(self.gwd_learner,
                                                                    self._buffer_min_size)
            else:
                self.gwd_learner.fit(self._user_ids,
                                     self._user_profiles,
                                     self._user_connections)
                # reset candidate
                self._ordered_cand_dict = {}

        # check if updating the GDL model is needed
        if self._only_init_learn:
            fit_weights, fit_groups = _consolidate_learned_info(self.gwd_learner,
                                                                self._buffer_min_size)
        else:
            current_iter = self._iter_counter
            update_period = self._update_iter_period
            if update_period is None:
                update_period = 1

            if current_iter % update_period == 0:
                # update distance metrics
                self.gwd_learner.fit(self._user_ids,
                                     self._user_profiles,
                                     self._user_connections)

                fit_weights, fit_groups = _consolidate_learned_info(self.gwd_learner,
                                                                    self._buffer_min_size)
            else:
                fit_weights, fit_groups = _consolidate_learned_info(self.gwd_learner,
                                                                    self._buffer_min_size)

        # update attribute
        # self._pdm_container = pdm_container
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

    def gen_suggestion(self, user_id, block_list=[]):
        """ generate recommendation list for target user: user_id
        """
        size = self._size

        # get a complete list of recommended user ordered
        # by distance
        if user_id in self._ordered_cand_dict:
            # retrieve the order commendation list
            sorted_cand_uids = self._ordered_cand_dict[user_id]
            # remove candidates in block_list
            if len(sorted_cand_uids) > size:
                suggestion = sorted_cand_uids[:size]
                del self._ordered_cand_dict[user_id][:size]
            else:
                suggestion = sorted_cand_uids
                self._ordered_cand_dict[user_id] = []

            # store recommended users
            if user_id in self._recommended_user_dict:
                self._recommended_user_dict[user_id].append(suggestion)
            else:
                self._recommended_user_dict[user_id] = suggestion

            return suggestion

        else:
            # first time to rank candidates since late update of GDL algorithm
            user_gid = self._return_user_group(user_id)
            # list all possible condidates
            cand_user_ids = self._user_ids
            # remove connected users
            block_list = self.get_connected_users(user_id)
            block_list.append(user_id)
            # retrieve recommended list
            if user_id in self._recommended_user_dict:
                recommended = self._recommended_user_dict[user_id]
                block_list.extend(recommended)

            cand_user_ids = [uid for uid in cand_user_ids if not uid in block_list]
            cand_user_dist = [None] * len(cand_user_ids)
            if len(cand_user_ids) > 0:
                for ii, cand_user_id in enumerate(cand_user_ids):
                    cand_user_dist[ii] = self._get_distance(user_id, cand_user_id)

            # sort candidates by distance
            sorted_list = sorted(zip(cand_user_ids, cand_user_dist), key=lambda pp: pp[1])
            sorted_cand_uids = [uid for uid, _ in sorted_list]

            # list in dicitonary is immutable
            # sorted_cand_uids is a pointer to
            # the element of a dicitonary
            suggestion = sorted_cand_uids[:size]
            del sorted_cand_uids[:size]

            # append the ordered list
            self._ordered_cand_dict[user_id] = sorted_cand_uids

            # append to ever recommended list
            if user_id in self._recommended_user_dict:
                self._recommended_user_dict[user_id].append(suggestion)
            else:
                self._recommended_user_dict[user_id] = suggestion

            return suggestion