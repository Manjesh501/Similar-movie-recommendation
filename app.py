import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Download NLTK resources
nltk.download('stopwords')

# Load dataset and preprocess
@st.cache_data
def load_data():
    df = pd.read_csv("movies.csv")
    
    # Data preprocessing
    df.drop_duplicates(subset=['title', 'release_date', 'runtime'], inplace=True)
    df = df[df.vote_count >= 20].reset_index(drop=True)
    df['genres'] = df['genres'].str.replace('-', ' ')
    df['keywords'] = df['keywords'].str.replace('-', ' ')
    df['credits'].fillna('', inplace=True)
    df['credits'] = df['credits'].apply(lambda x: ' '.join(x.replace(' ', '').split('-')[:5]))
    df['tags'] = df['overview'] + ' ' + df['genres'] + ' ' + df['keywords'] + ' ' + df['credits'] + ' ' + df['original_language']
    df['tags'].fillna('', inplace=True)
    df['tags'] = df['tags'].str.replace('[^\w\s]', '')

    # Stemming
    stemmer = SnowballStemmer("english")
    df['tags'] = df['tags'].apply(lambda x: ' '.join(stemmer.stem(i) for i in x.split()))

    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['tags'])
    
    return df, tfidf_matrix

df, tfidf_matrix = load_data()

# Movie recommendations based on cosine similarity
@st.cache_resource
def get_recommendations_by_preferences(genres_preference, mood_preference, time_available):
    # Movie filtering based on user preferences (genres, mood, time)
    df_filtered = df.copy()
    if genres_preference:
        df_filtered = df_filtered[df_filtered['genres'].str.contains('|'.join(genres_preference), case=False, na=False)]
    if mood_preference:
        mood_keywords = {"Lighthearted": "comedy|feel good", "Adventurous": "action|adventure", "Romantic": "romance", "Thriller": "thriller"}
        df_filtered = df_filtered[df_filtered['tags'].str.contains(mood_keywords.get(mood_preference, ""), case=False, na=False)]
    if time_available:
        df_filtered = df_filtered[df_filtered['runtime'] <= time_available]

    return df_filtered.head(15)  # Limit to 15 recommendations

# Refactored function to get recommendations by movie title
@st.cache_resource
def get_recommendations_by_movie(title):
    try:
        idx = df.index[df['title'].str.lower() == title.lower()][0]
    except IndexError:
        st.write(f"Movie '{title}' not found in the dataset.")
        return None

    sim_scores = list(enumerate(cosine_similarity(tfidf_matrix[idx], tfidf_matrix)[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:16]  # Exclude the selected movie itself, limit to 15 recommendations
    movie_indices = [i[0] for i in sim_scores]
    
    return df.iloc[movie_indices]

# Streamlit app layout
st.title('🎬 Personalized Movie Recommendation System')

# Choose recommendation type
recommendation_type = st.radio("Select Recommendation Type", options=["By Preferences", "By Similar Movie", "By Watch History/Favorite Movies"])

if recommendation_type == "By Preferences":
    # Collect user preferences
    genres_preference = st.multiselect("Select your favorite genres", options=["Action", "Comedy", "Drama", "Science Fiction", "Romantic", "Thriller", "Adventure"])
    mood_preference = st.selectbox("What's your mood?", options=["Lighthearted", "Adventurous", "Romantic", "Thriller", "Inspiring"])
    time_available = st.slider("Time available (in minutes)", min_value=30, max_value=240, step=15)

    with st.expander("More details"):
        st.write("Use the preferences above to refine movie recommendations.")
    
    # Get and display movie recommendations
    if st.button('Recommend'):
        recommendations = get_recommendations_by_preferences(genres_preference, mood_preference, time_available)
        if recommendations.empty:
            st.write("Sorry, no recommendations found based on your preferences.")
        else:
            st.write("Here are some personalized movie recommendations based on your preferences:")
            cols = st.columns(4)
            for idx, movie in recommendations.iterrows():
                img_url = f'https://image.tmdb.org/t/p/w500/{movie["poster_path"]}'
                with cols[idx % 4]:
                    st.image(img_url, caption=movie['title'], use_column_width=True)

elif recommendation_type == "By Similar Movie":
    # Provide a list of available movies to select from
    movie_titles = df['title'].tolist()
    favorite_movie = st.selectbox("Choose a movie", movie_titles)
    
    with st.expander("More details"):
        st.write("This option finds movies similar to your selected favorite movie.")

    if st.button('Recommend Similar Movies'):
        recommendations = get_recommendations_by_movie(favorite_movie)
        if recommendations is not None:
            st.write("Here are some movies similar to your favorite movie:")
            cols = st.columns(4)
            for idx, movie in recommendations.iterrows():
                img_url = f'https://image.tmdb.org/t/p/w500/{movie["poster_path"]}'
                with cols[idx % 4]:
                    st.image(img_url, caption=movie['title'], use_column_width=True)

elif recommendation_type == "By Watch History/Favorite Movies":
    favorite_movies = st.multiselect("Select your favorite movies", options=df['title'].tolist())

    with st.expander("More details"):
        st.write("Select your favorite movies to get recommendations based on your movie preferences.")

    if st.button('Recommend Based on Favorites'):
        all_recommendations = []
        for movie in favorite_movies:
            recommendations = get_recommendations_by_movie(movie)
            if recommendations is not None:
                all_recommendations.append(recommendations)

        if all_recommendations:
            combined_recommendations = pd.concat(all_recommendations).drop_duplicates(subset=['title'])
            combined_recommendations = combined_recommendations.sample(frac=1).head(15)
            st.write("Here are some movie recommendations based on your watch history:")
            cols = st.columns(4)
            for idx, movie in combined_recommendations.iterrows():
                img_url = f'https://image.tmdb.org/t/p/w500/{movie["poster_path"]}'
                with cols[idx % 4]:
                    st.image(img_url, caption=movie['title'], use_column_width=True)
        else:
            st.warning("Please select at least one favorite movie to get recommendations.")
