# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 08:34:27 2024

@author: ketan
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

#load the CSV file
file_path="C:/10-Recommendation engine/game.csv.xls"
data=pd.read_csv(file_path)

#setp 1: Create a user-item matrix
user_item_matrix=data.pivot_table(index='userId',columns='game',values='rating')

'''
pivot_table:This function reshape the DataFrame into matrix where:
    Each row represent a user (identifed by userId)
    Each Columns represnt a game(identifed by Game)
    The value in the martrix represeent the rating that user gave to the game.
'''

#step 2: Fill NaN values with 0
user_item_matrix_filled=user_item_matrix.fillna(0)
'''
This line replaces any missing values (NaNs)
in the user-item matrix with 0
indicating that user did not rate the particular game.
'''

#step 3: Compute the cosine similatry between user based on raw ratings
user_similarity=cosine_similarity(user_item_matrix_filled)

#convert similarity matrix to a DataFrame for easy reference
user_similarity_df=pd.DataFrame(user_similarity,index=user_item_matrix.index,columns=user_item_matrix.index)

# step4: function to get game recommendation for a specific user
def get_collaborative_recommendations_for_user(user_id,num_recommendations=5):
    # ger similarity
    similar_users=user_similarity_df[user_id].sort_values(ascending=False)
    
    # get most similar user 
    similar_users=similar_users.drop(user_id)
    
    # select the top N similar users to limit noise
    top_similar_users=similar_users.head(50)
    # this selects the top 50 most similar users to limit noise in the recommendation 
    #get rating of these similar users, weighted by their similarity score
    weighted_ratings=np.dot(top_similar_users.values,user_item_matrix_filled.loc[top_similar_users.index])
    
    # np.dot() : this computes the dot product bet. the similarity
    # scores of the top similar users and their corresponding ratings in the 
    # user item matrix.
    # the result is an array of weighted ratings for each game.
    # Noramalize by the sum of similarities
    
    sum_of_similarities=top_similar_users.sum()
    
    if sum_of_similarities>0:
        weighted_ratings/=sum_of_similarities
    
    # the weighted ratings 
    
    
    
    user_ratings=user_item_matrix_filled.loc[user_id]
    unrated_games=user_ratings[user_ratings==0]
    #identifies games that target user has not rated
    
    # get the weighted scores for unrated games
    game_recommendations=pd.Series(weighted_ratings,index=user_item_matrix_filled.columns).loc[unrated_games.index]

# this creates a pandas series from the weighted ratings
# and filters it to include only the unrated games 
# finally, it sorts the recommendation in descending order
# and returns the top specified number of recommendations.

# returns the top 'num-recommendations' game recommendations
    return game_recommendations.sort_values(ascending=False).head(num_recommendations)
    
#  Example usage : get recommendations for a user with ID 3
recommendation_games=get_collaborative_recommendations_for_user(user_id=3)

# print the recommended games
print("Recommended games for user 3:")
print(recommendation_games)   

#############################################################
   
#  Example usage : get recommendations for a user with ID 1087
recommendation_games=get_collaborative_recommendations_for_user(user_id=1087)

# print the recommended games
print("Recommended games for user 1087:")
print(recommendation_games)       
    
###############################################################

#  Example usage : get recommendations for a user with ID 2614
recommendation_games=get_collaborative_recommendations_for_user(user_id=2614)

# print the recommended games
print("Recommended games for user 2614:")
print(recommendation_games)       

##############################################################
   
#  Example usage : get recommendations for a user with ID 3
recommendation_games=get_collaborative_recommendations_for_user(user_id=22)

# print the recommended games
print("Recommended games for user 22:")
print(recommendation_games)



