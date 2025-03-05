import streamlit as st
import pandas as pd
import sys
import os

# Page configuration MUST be the first Streamlit command
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.recommendation import get_recommendations_by_movie, get_recommendations_by_preferences
from src.data_loader import preprocess_data
from src.utils import get_poster_url

# Custom CSS
st.markdown("""
    <style>
    .movie-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 1rem;
        padding: 1rem;
    }
    .movie-card {
        border-radius: 10px;
        padding: 10px;
        background: #f0f2f6;
        text-align: center;
    }
    .movie-title {
        margin-top: 8px;
        font-size: 1rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Load data
df = preprocess_data()

# Streamlit UI
st.title('🎬 Personalized Movie Recommendation System')
st.markdown("---")

recommendation_type = st.radio("Select Recommendation Type", ["By Preferences", "By Similar Movie"])

if recommendation_type == "By Preferences":
    col1, col2, col3 = st.columns([2,1,1])
    
    with col1:
        genres = st.multiselect(
            "Select genres",
            options=["Action", "Comedy", "Drama", "Science Fiction", "Romantic", "Thriller", "Adventure"]
        )
    with col2:
        mood = st.selectbox(
            "Your mood?",
            ["Lighthearted", "Adventurous", "Romantic", "Thriller", "Inspiring"]
        )
    with col3:
        time_available = st.slider(
            "Available time (minutes)",
            min_value=30,
            max_value=240,
            step=15
        )

    if st.button("Recommend", type="primary"):
        recommendations = get_recommendations_by_preferences(genres, mood, time_available)
        if recommendations.empty:
            st.error("No recommendations found!")
        else:
            st.subheader("📌 Recommended Movies")
            st.markdown("---")
            
            # Display movies in a grid
            for i in range(0, len(recommendations), 4):
                cols = st.columns(4)
                for j, col in enumerate(cols):
                    if i + j < len(recommendations):
                        movie = recommendations.iloc[i + j]
                        with col:
                            st.image(
                                get_poster_url(movie['poster_path']),
                                caption=movie['title'],
                                use_container_width=True
                            )

elif recommendation_type == "By Similar Movie":
    movie_titles = sorted(df['title'].tolist())
    favorite_movie = st.selectbox("Choose a movie", movie_titles)

    if st.button("Recommend Similar Movies", type="primary"):
        recommendations = get_recommendations_by_movie(favorite_movie)
        if recommendations is not None:
            st.subheader("📌 Similar Movies")
            st.markdown("---")
            
            # Display movies in a grid
            for i in range(0, len(recommendations), 4):
                cols = st.columns(4)
                for j, col in enumerate(cols):
                    if i + j < len(recommendations):
                        movie = recommendations.iloc[i + j]
                        with col:
                            st.image(
                                get_poster_url(movie['poster_path']),
                                caption=movie['title'],
                                use_container_width=True
                            )
        else:
            st.error("No similar movies found!")
