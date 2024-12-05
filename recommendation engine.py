# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#load the dataset
file_path="C:/10-Recommendation engine/Entertainment.csv.xls"
data =pd.read_csv(file_path)

#setp 1:preprocess the 'category' columns using TF-IDF
tfidf=TfidfVectorizer(stop_words='english') #Remove the stopwords
tfidf_matrix=tfidf.fit_transform(data['Category']) #fit and transform the category data

#Setp 2:compute the cosine similarity between title
cosine_sim=cosine_similarity(tfidf_matrix, tfidf_matrix)

#step 3: create a function to recommend titles on similarity
def get_recommendations(title,cosine_sim=cosine_sim):
    #Get the index of title that matches the input title
    idx=data[data['Titles']==title].index[0]
    '''
    data=['Title']==title
    see the photo
    '''
    #Get the pairwise similarity score of all title with that title
    sim_scores=list(enumerate(cosine_sim[idx]))
    
    #sort the title based on the similarity score in descending on
    sim_scores=sorted(sim_scores,key=lambda x:x[1],reverse=True)
    
    #Get the indices of most similar titles
    sim_indices=[i[0] for i in sim_scores[1:6]]
    
    #return the top 5 most similar title
    return data['Title'].iloc[sim_indices]

#test the recommendation system with examole title
example_title="Toy Story (1995)"
recommended_title =get_recommendations(example_title)

#print the recommendation
print(f"Recommendation for'{example_title}")
for title in recommended_title: 
    print(title)