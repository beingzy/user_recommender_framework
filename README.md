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

 - UserRecommenderMixin (UserRecommender.py)
 - NNUserRecommender (NNUserRecommender.py)
 - PairwiseDistMatrix (PairwiseDistMatrix.py)

## Simulation Framework to evaluate the performance of user recommendation system 
(in sub-module: network_simulator)
  * UserRecSysExpSimulator
  * UserClickSimulator
  * SocialNetworkEvaluator

