"""
models/content_based.py
Content-Based Filtering using TF-IDF + cosine similarity on product features.
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


class ContentBasedFilter:
    """
    Recommends products similar to those a user has rated highly.
    Features: category, brand, price_bucket, description.
    """

    def __init__(self):
        self.tfidf         = TfidfVectorizer(stop_words="english", max_features=500)
        self.scaler        = MinMaxScaler()
        self.sim_matrix    = None          # (n_products, n_products)
        self.products_df   = None
        self.product_index = {}            # product_id → row index

    # ── Feature Engineering ───────────────────────────────────────────────────
    def _build_features(self, products_df: pd.DataFrame) -> np.ndarray:
        # Numeric: price scaled
        prices = self.scaler.fit_transform(
            products_df[["price"]].fillna(products_df["price"].median())
        )

        # Text: category + brand + description
        text = (
            products_df["category"].fillna("") + " "
            + products_df["brand"].fillna("") + " "
            + products_df["description"].fillna("")
        )
        tfidf_matrix = self.tfidf.fit_transform(text).toarray()

        return np.hstack([tfidf_matrix, prices])

    # ── Training ──────────────────────────────────────────────────────────────
    def fit(self, products_df: pd.DataFrame):
        self.products_df = products_df.reset_index(drop=True)
        self.product_index = {
            pid: idx
            for idx, pid in enumerate(self.products_df["product_id"])
        }

        features = self._build_features(self.products_df)
        self.sim_matrix = cosine_similarity(features)
        print(f"✅ CBF fitted. Similarity matrix: {self.sim_matrix.shape}")

    # ── Similar Products ──────────────────────────────────────────────────────
    def similar_products(self, product_id: int, n: int = 10) -> list[dict]:
        if product_id not in self.product_index:
            return []
        idx   = self.product_index[product_id]
        scores = list(enumerate(self.sim_matrix[idx]))
        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for i, score in scores[1:n+1]:
            row = self.products_df.iloc[i]
            results.append({
                "product_id":   int(row["product_id"]),
                "product_name": row["product_name"],
                "category":     row["category"],
                "similarity":   round(float(score), 4),
            })
        return results

    # ── User Recommendation (profile-based) ───────────────────────────────────
    def recommend(self, rated_product_ids: list[int],
                  rated_scores: list[float],
                  n: int = 10,
                  exclude_ids: list[int] = None) -> list[dict]:
        """
        Build a weighted user profile from rated products → rank all others.
        """
        if not rated_product_ids:
            return []

        exclude = set(exclude_ids or rated_product_ids)
        n_prods = len(self.products_df)

        # Weighted average of rated product feature vectors
        feature_matrix = self._build_features(self.products_df)
        profile = np.zeros(feature_matrix.shape[1])
        total_w = 0.0

        for pid, score in zip(rated_product_ids, rated_scores):
            if pid in self.product_index:
                idx      = self.product_index[pid]
                weight   = (score - 1) / 4.0          # normalise 1-5 → 0-1
                profile += weight * feature_matrix[idx]
                total_w += weight

        if total_w == 0:
            return []

        profile /= total_w
        sims    = cosine_similarity([profile], feature_matrix)[0]

        ranked = sorted(
            [(i, s) for i, s in enumerate(sims)
             if int(self.products_df.iloc[i]["product_id"]) not in exclude],
            key=lambda x: x[1], reverse=True
        )

        results = []
        for i, score in ranked[:n]:
            row = self.products_df.iloc[i]
            results.append({
                "product_id":   int(row["product_id"]),
                "product_name": row["product_name"],
                "category":     row["category"],
                "price":        float(row["price"]),
                "similarity":   round(float(score), 4),
            })
        return results

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self, path: str = "models/cbf_model.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"💾 CBF model saved → {path}")

    @staticmethod
    def load(path: str = "models/cbf_model.pkl") -> "ContentBasedFilter":
        with open(path, "rb") as f:
            return pickle.load(f)
