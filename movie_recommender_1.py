import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#helper functions
def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]


#read csv
df = pd.read_csv("movie_dataset.csv")
print(df.head())
print(df.columns)

#select features
features = ['keywords', 'cast', 'genres', 'director']

#combining all the selected features in new column
for feature in features:
    df[feature] = df[feature].fillna(" ")
    
def combine_features(row):
        return row['keywords'] + " " + row['cast'] + " " + row['genres'] + " " + row['director']

df["combined_features"] = df.apply(combine_features, axis=1) 


#create count matrix from this combined column
cv = CountVectorizer()
count_matrix = cv.fit_transform(df['combined_features'])

#compute cosine similarity
cosine_sim = cosine_similarity(count_matrix)

movie_user_likes = "Avatar"

#get movie index from its title
movie_index = get_index_from_title(movie_user_likes)
similar_movies = list(enumerate(cosine_sim[movie_index]))

#sort in descendeding order
sorted_similar_movies = sorted(similar_movies, key=lambda x:x[1], reverse=True)

#print title of first 50 movies
i=0
for movie in sorted_similar_movies:
    print(get_title_from_index(movie[0]))
    i=i+1
    if(i>50):
        break



