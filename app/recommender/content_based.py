"""
Content-based filtering.
Builds a TF-IDF profile per restaurant from (cuisine + tags), builds a user profile
as the weighted average of the TF-IDF vectors of restaurants the user rated highly,
then scores unseen restaurants by cosine similarity to that profile.
Also applies a haversine-distance filter/boost for location relevance.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from math import radians, sin, cos, sqrt, atan2


class ContentBasedRecommender:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.restaurant_ids = []
        self.tfidf_matrix = None
        self.restaurants_df = None

    def fit(self, restaurants_df: pd.DataFrame):
        """
        restaurants_df columns: id, name, cuisine, tags, price_range, latitude, longitude
        """
        self.restaurants_df = restaurants_df.reset_index(drop=True)
        corpus = (restaurants_df["cuisine"].fillna("") + " " + restaurants_df["tags"].fillna(""))
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.restaurant_ids = restaurants_df["id"].tolist()

    def _user_profile(self, rated_restaurant_ids: list, weights: list):
        idx_map = {rid: i for i, rid in enumerate(self.restaurant_ids)}
        rows = [idx_map[rid] for rid in rated_restaurant_ids if rid in idx_map]
        if not rows:
            return None
        w = np.array([weights[i] for i, rid in enumerate(rated_restaurant_ids) if rid in idx_map])
        vectors = self.tfidf_matrix[rows]
        w = w / w.sum()
        profile = np.asarray(vectors.T.dot(w)).flatten()
        return profile

    def score_all(self, rated_restaurant_ids: list, ratings: list) -> dict:
        """Returns cosine-similarity score for every restaurant given a user's rating history."""
        if not rated_restaurant_ids:
            return {rid: 0.0 for rid in self.restaurant_ids}

        profile = self._user_profile(rated_restaurant_ids, ratings)
        if profile is None:
            return {rid: 0.0 for rid in self.restaurant_ids}

        sims = cosine_similarity(self.tfidf_matrix, profile.reshape(1, -1)).flatten()
        return dict(zip(self.restaurant_ids, sims))

    @staticmethod
    def haversine_km(lat1, lon1, lat2, lon2):
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        return 2 * R * atan2(sqrt(a), sqrt(1 - a))

    def distance_scores(self, user_lat: float, user_lon: float, max_km: float = 10.0) -> dict:
        """1.0 = right next to user, 0.0 = at/over max_km away."""
        scores = {}
        for _, row in self.restaurants_df.iterrows():
            if user_lat is None or user_lon is None or row["latitude"] is None:
                scores[row["id"]] = 0.5  # neutral if location missing
                continue
            d = self.haversine_km(user_lat, user_lon, row["latitude"], row["longitude"])
            scores[row["id"]] = max(0.0, 1 - d / max_km)
        return scores
