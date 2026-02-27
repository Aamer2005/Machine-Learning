import numpy as np
import pandas as pd
import difflib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import streamlit as st

#load dataset
movies_data = pd.read_csv("movies.csv")

#selecting the relevant features for recommendation

selected_features = ["genres","keywords","tagline","cast","director"]

for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']

# converting the text data to feature vectors

vetcorizer = TfidfVectorizer()

feature_vectors = vetcorizer.fit_transform(combined_features)

#cosine similarity
similarity = cosine_similarity(feature_vectors)

list_of_titles = movies_data['title'].tolist()

# Streamlit UI
st.title("🎬 Movie Recommendation System")

movie_name = st.text_input("Enter your favourite movie")

if st.button("Recommend"):
    find_close_match = difflib.get_close_matches(movie_name, list_of_titles)

    if len(find_close_match) == 0:
        st.write("Movie not found")
    else:
        close_match = find_close_match[0]
        index = movies_data[movies_data.title == close_match].index[0]
        similarity_score = list(enumerate(similarity[index]))

        sorted_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

        st.subheader("Recommended Movies:")
        for i, movie in enumerate(sorted_movies[1:11]):
            movie_index = movie[0]
            title = movies_data.iloc[movie_index]['title']
            st.write(i+1, title)