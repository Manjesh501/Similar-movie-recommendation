import pandas as pd
import os

from sklearn.metrics.pairwise import cosine_similarity
from src.data_loader import preprocess_data, get_tfidf_matrix
# Initialize the data
df = preprocess_data()
tfidf_matrix = get_tfidf_matrix(df)

def get_recommendations_by_movie(title):
    """Get movie recommendations based on a similar movie."""
    try:
        idx = df.index[df['title'].str.lower() == title.lower()][0]
    except IndexError:
        return None  # Movie not found

    sim_scores = list(enumerate(cosine_similarity(tfidf_matrix[idx], tfidf_matrix)[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:16]
    
    return df.iloc[[i[0] for i in sim_scores]]

def get_recommendations_by_preferences(genres, mood, runtime):
    df_filtered = df.copy()
    if genres:
        df_filtered = df_filtered[df_filtered['genres'].str.contains('|'.join(genres), case=False, na=False)]
    if mood:
        mood_keywords = {"Lighthearted": "comedy|feel good", "Adventurous": "action|adventure", "Romantic": "romance", "Thriller": "thriller"}
        df_filtered = df_filtered[df_filtered['tags'].str.contains(mood_keywords.get(mood, ""), case=False, na=False)]
    if runtime:
        df_filtered = df_filtered[df_filtered['runtime'] <= runtime]

    return df_filtered.head(15)
