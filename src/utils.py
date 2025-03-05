from src.config import POSTER_URL

def get_poster_url(poster_path):
    return f"{POSTER_URL}{poster_path}" if poster_path else None
