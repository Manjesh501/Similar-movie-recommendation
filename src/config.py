import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'movies.csv')  # Update 'movies.csv' to match your actual filename
POSTER_URL = "https://image.tmdb.org/t/p/w500/"