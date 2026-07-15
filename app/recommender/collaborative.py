"""
Collaborative filtering using matrix factorization (SGD-trained latent-factor
model, the same idea as Surprise's SVD) implemented in plain NumPy.
No compiled/Cython dependency, so it installs cleanly everywhere (incl. Windows
without a C++ compiler).

Model: rating(u, i) ~= global_mean + bias_u + bias_i + p_u . q_i
Trained with stochastic gradient descent + L2 regularization.
"""
import numpy as np
import pandas as pd


class CollaborativeRecommender:
    def __init__(self, n_factors=50, n_epochs=20, lr=0.005, reg=0.02, random_state=42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.rng = np.random.default_rng(random_state)

        self.trained = False
        self.global_mean = 3.0
        self.user_index = {}
        self.item_index = {}
        self.P = None
        self.Q = None
        self.bu = None
        self.bi = None

    def fit(self, ratings_df: pd.DataFrame):
        if ratings_df.empty:
            self.trained = False
            return

        users = ratings_df["user_id"].unique()
        items = ratings_df["restaurant_id"].unique()
        self.user_index = {u: i for i, u in enumerate(users)}
        self.item_index = {m: i for i, m in enumerate(items)}

        n_users, n_items = len(users), len(items)
        self.global_mean = ratings_df["rating"].mean()

        self.P = self.rng.normal(0, 0.1, (n_users, self.n_factors))
        self.Q = self.rng.normal(0, 0.1, (n_items, self.n_factors))
        self.bu = np.zeros(n_users)
        self.bi = np.zeros(n_items)

        rows = list(zip(
            ratings_df["user_id"].map(self.user_index),
            ratings_df["restaurant_id"].map(self.item_index),
            ratings_df["rating"],
        ))

        for _ in range(self.n_epochs):
            self.rng.shuffle(rows)
            for u, i, r in rows:
                pred = self.global_mean + self.bu[u] + self.bi[i] + self.P[u] @ self.Q[i]
                err = r - pred

                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])

                p_u = self.P[u].copy()
                self.P[u] += self.lr * (err * self.Q[i] - self.reg * p_u)
                self.Q[i] += self.lr * (err * p_u - self.reg * self.Q[i])

        self.trained = True

    def predict(self, user_id, restaurant_id) -> float:
        if not self.trained:
            return self.global_mean

        u = self.user_index.get(user_id)
        i = self.item_index.get(restaurant_id)

        if u is None or i is None:
            bu = self.bu[u] if u is not None else 0.0
            bi = self.bi[i] if i is not None else 0.0
            return float(np.clip(self.global_mean + bu + bi, 1, 5))

        pred = self.global_mean + self.bu[u] + self.bi[i] + self.P[u] @ self.Q[i]
        return float(np.clip(pred, 1, 5))

    def predict_batch(self, user_id, restaurant_ids: list) -> dict:
        return {rid: self.predict(user_id, rid) for rid in restaurant_ids}