## Framework for Developing User Recommendation System (in sub-module: user_recommender)
The module is designed to support the prototype development and evaluation of user recommendation
 system. The current version includes following algorithm:

  * Nearest Neigbour User Recommendation System (Distance-based): utilize the user profile information only, with
customized or generic distance metrics.

  * Clustering-based User Recommendation System: utilize the user profile information only, with clustering algorithm
to improve the efficiency of searching suggestions.

  * Learning Distance Metrics-based User Recommendation System: utilize both user profile information and existing
social connections, requiring more computational resource.

## Existing Objects in Module

 - UserRecommenderMixin 
 - NNUserRecommender 
 - PairwiseDistMatrix

## How to use experiemntation framework to evaluate recommendation system
1. Conduct an experiment of applying Nearest Neigbour-based recommendation system:
```python
from user_recommender_framework.network_simulator import *
from user_recommender_framework.user_recommender import *

# load data for experimentation
# 1. user_ids: all users ids
# 2. user_profiles: user profiles
# 3. user_connections: list of user connections
# 4. init_user_connections: sample user connections which experiment starts with
# ...

# prepare components for experiment
nnu_recommender = NNUserRecommender(user_ids, user_profiles, init_user_connections)

evaluator = SocialNetworkEvaluator()
evaluator.load_ref_user_conncetions(user_connections)

# user behavior simulator 
user_clicker = UserClickSimulator()

# setup experiment
experimentor = UserRecSysExpSimulator(name="MyExperiment")
experimentor.load_recommender(nuu_recommender)
experimentor.load_evaluator(evaluator)
experimentor.load_clicker(user_clicker)

# set the number of suggestions for each user at each iteration
experimentor.set_recommendation_size(5)
# start experiemnt, expriement results will be exported automatically
experimentor.run()

# to retore experiment to status before .run()
experimentor.reset()
```

