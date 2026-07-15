"""
Hybrid recommender: weighted blend of
  - collaborative filtering score (normalized 0-1 from the 1-5 SVD prediction)
  - content-based cosine similarity score (0-1)
  - location proximity score (0-1)

final_score = w_cf * cf + w_cb * cb + w_loc * loc

Cold-start handling:
  - New user (no ratings)      -> weight shifts fully to content-based + location
  - New restaurant (no ratings)-> still scored via content-based + location
"""
import pandas as pd
from app.recommender.collaborative import CollaborativeRecommender
from app.recommender.content_based import ContentBasedRecommender


class HybridRecommender:
    def __init__(self, w_cf=0.5, w_cb=0.3, w_loc=0.2):
        self.cf = CollaborativeRecommender()
        self.cb = ContentBasedRecommender()
        self.w_cf = w_cf
        self.w_cb = w_cb
        self.w_loc = w_loc

    def fit(self, ratings_df: pd.DataFrame, restaurants_df: pd.DataFrame):
        self.cf.fit(ratings_df)
        self.cb.fit(restaurants_df)
        self.restaurants_df = restaurants_df

    def recommend(self, user_id: int, user_lat: float, user_lon: float,
                  rated_restaurant_ids: list, rated_values: list,
                  top_n: int = 10, max_km: float = 10.0):

        all_ids = self.restaurants_df["id"].tolist()
        unseen_ids = [rid for rid in all_ids if rid not in rated_restaurant_ids]

        # 1. Collaborative scores, normalized from [1,5] -> [0,1]
        cf_raw = self.cf.predict_batch(user_id, unseen_ids)
        cf_scores = {rid: (v - 1) / 4 for rid, v in cf_raw.items()}

        # 2. Content-based scores
        cb_scores = self.cb.score_all(rated_restaurant_ids, rated_values)

        # 3. Location scores
        loc_scores = self.cb.distance_scores(user_lat, user_lon, max_km=max_km)

        # cold-start: no rating history -> drop collaborative weight to 0
        w_cf, w_cb, w_loc = self.w_cf, self.w_cb, self.w_loc
        if not rated_restaurant_ids:
            total = w_cb + w_loc
            w_cf, w_cb, w_loc = 0.0, w_cb / total, w_loc / total

        results = []
        for rid in unseen_ids:
            score = (w_cf * cf_scores.get(rid, 0.0)
                     + w_cb * cb_scores.get(rid, 0.0)
                     + w_loc * loc_scores.get(rid, 0.5))
            results.append((rid, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]
