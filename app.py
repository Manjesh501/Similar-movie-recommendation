import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from skimage import io
import streamlit as st

@st.cache_data
def load_data():
    df = pd.read_csv("movies.csv")
    
    # Data preprocessing
    df.drop_duplicates(subset=['title', 'release_date', 'runtime'], inplace=True)
    df = df[df.vote_count >= 20].reset_index()
    df['genres'] = df['genres'].str.replace('-', ' ')
    df['keywords'] = df['keywords'].str.replace('-', ' ')
    df['credits'].fillna('', inplace=True)
    df['credits'] = df['credits'].apply(lambda x: ' '.join(x.replace(' ', '').split('-')[:5]))
    df['tags'] = df['overview'] + ' ' + df['genres'] + ' ' + df['keywords'] + ' ' + df['credits'] + ' ' + df['original_language']
    df['tags'].fillna('', inplace=True)
    df['tags'] = df['tags'].str.replace('[^\w\s]', '')

    stemmer = SnowballStemmer("english")
    df['tags'] = df['tags'].apply(lambda x: ' '.join(stemmer.stem(i) for i in x.split()))

    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['tags'])
    
    return df, tfidf_matrix

df, tfidf_matrix = load_data()

@st.cache_resource
def get_recommendations(title):
    lower_title = title.lower()
    try:
        idx = df.index[df['title'].str.lower() == lower_title][0]
    except IndexError:
        st.write(f"Movie '{title}' not found in the dataset.")
        return None

    try:
        a = io.imread(f'https://image.tmdb.org/t/p/w500/{df.loc[idx, "poster_path"]}')
        st.image(a, caption=title, width=200)
    except:
        pass
    
    st.write('Recommendations:')

    sim_scores = list(enumerate(cosine_similarity(tfidf_matrix, tfidf_matrix[idx])))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[1] > 0]  # Filter out zero similarity
    sim_scores = sim_scores[1:10]
    movie_indices = [i[0] for i in sim_scores]
    result = df.iloc[movie_indices]

    num_cols = 4
    rows = (len(result) // num_cols) + 1
    for row in range(rows):
        cols = st.columns(num_cols)
        for col_idx in range(num_cols):
            movie_idx = row * num_cols + col_idx
            if movie_idx < len(result):
                with cols[col_idx]:
                    movie = result.iloc[movie_idx]
                    try:
                        poster_path = movie["poster_path"]
                        if poster_path:
                            img_url = f'https://image.tmdb.org/t/p/w500/{poster_path}'
                            st.image(img_url, caption=movie['title'], use_column_width=True)
                        else:
                            st.write(movie['title'])
                    except:
                        st.write(movie['title'])

st.title('Movie Recommendation System')

# Autocomplete for movie titles
movie_titles = df['title'].tolist()

selected_movie = st.selectbox('Type or select a movie from the dropdown', movie_titles)

if st.button('Recommend'):
    get_recommendations(selected_movie)

st.sidebar.header('Additional Options')
if st.sidebar.checkbox('Show Dataset'):
    st.write(df)
