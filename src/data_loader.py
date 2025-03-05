import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import os

from nltk.stem.snowball import SnowballStemmer
from src.config import DATA_PATH

nltk.download('stopwords')

def load_data():
    """Load the movie dataset."""
    try:
        if not os.path.exists(DATA_PATH):
            st.error(f"Dataset not found at {DATA_PATH}")
            st.stop()
        return pd.read_csv(DATA_PATH)
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        st.stop()

@st.cache_resource
def preprocess_data():
    """Preprocess the movie dataset."""
    df = load_data()
    df = df.copy()  # Create a copy to avoid SettingWithCopyWarning
    df.drop_duplicates(subset=['title', 'release_date', 'runtime'], inplace=True)
    df = df[df.vote_count >= 20].reset_index(drop=True)
    
    # Process string columns
    df.loc[:, 'genres'] = df['genres'].str.replace('-', ' ')
    df.loc[:, 'keywords'] = df['keywords'].str.replace('-', ' ')
    df.loc[:, 'credits'] = df['credits'].fillna('')
    df.loc[:, 'credits'] = df['credits'].apply(lambda x: ' '.join(x.replace(' ', '').split('-')[:5]))
    
    # Create tags
    df.loc[:, 'tags'] = (df['overview'].fillna('') + ' ' + 
                        df['genres'].fillna('') + ' ' + 
                        df['keywords'].fillna('') + ' ' + 
                        df['credits'] + ' ' + 
                        df['original_language'].fillna(''))
    
    df.loc[:, 'tags'] = df['tags'].str.replace('[^\w\s]', '')
    
    # Stemming
    stemmer = SnowballStemmer("english")
    df.loc[:, 'tags'] = df['tags'].apply(lambda x: ' '.join(stemmer.stem(i) for i in x.split()))
    
    return df

@st.cache_resource
def get_tfidf_matrix(df):
    """Generate TF-IDF matrix for movie tags."""
    tfidf = TfidfVectorizer(stop_words='english')
    return tfidf.fit_transform(df['tags'])

# Remove global initialization
__all__ = ['preprocess_data', 'get_tfidf_matrix', 'load_data']
