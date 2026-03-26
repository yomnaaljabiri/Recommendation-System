"""
models/hybrid.py
Hybrid recommender: weighted ensemble of CF + CBF + NeuMF.
Falls back gracefully when a component is unavailable.
"""
import pickle
import numpy as np
import pandas as pd


class HybridRecommender:
    """
    Score fusion: final_score = w_cf*cf + w_cbf*cbf + w_ncf*ncf
    Weights are auto-normalised.
    """

    def __init__(self, cf_model=None, cbf_model=None, ncf_model=None,
                 weights=(0.4, 0.3, 0.3)):
        self.cf   = cf_model
        self.cbf  = cbf_model
        self.ncf  = ncf_model
        w         = np.array(weights, dtype=float)
        self.w    = w / w.sum()             # normalise
        self.products_df = None

    def set_products(self, products_df: pd.DataFrame):
        self.products_df = products_df.set_index("product_id")

    # ── Core recommendation ────────────────────────────────────────────────────
    def recommend(self, user_id: int, n: int = 10,
                  rated_products: list = None,
                  rated_scores:   list = None) -> list[dict]:

        rated_ids = list(rated_products or [])
        scores_map: dict[int, float] = {}

        # --- CF ---
        if self.cf and self.cf._trained:
            for r in self.cf.recommend(user_id, n=50, rated_products=rated_ids):
                pid = r["product_id"]
                s   = (r["predicted_rating"] - 1) / 4.0   # normalise 1-5 → 0-1
                scores_map[pid] = scores_map.get(pid, 0) + self.w[0] * s

        # --- CBF ---
        if self.cbf and rated_ids:
            cbf_recs = self.cbf.recommend(
                rated_product_ids=rated_ids,
                rated_scores=rated_scores or [3.0]*len(rated_ids),
                n=50, exclude_ids=rated_ids
            )
            for r in cbf_recs:
                pid = r["product_id"]
                scores_map[pid] = scores_map.get(pid, 0) + self.w[1] * r["similarity"]

        # --- NeuMF ---
        if self.ncf and self.ncf._trained:
            for r in self.ncf.recommend(user_id, n=50, rated_products=rated_ids):
                pid = r["product_id"]
                s   = (r["predicted_rating"] - 1) / 4.0
                scores_map[pid] = scores_map.get(pid, 0) + self.w[2] * s

        # Sort & enrich
        ranked = sorted(scores_map.items(), key=lambda x: x[1], reverse=True)[:n]
        results = []
        for pid, score in ranked:
            info = {}
            if self.products_df is not None and pid in self.products_df.index:
                row  = self.products_df.loc[pid]
                info = {
                    "product_name": row.get("product_name", ""),
                    "category":     row.get("category",     ""),
                    "price":        round(float(row.get("price", 0)), 2),
                    "avg_rating":   float(row.get("avg_rating", 0)),
                    "brand":        row.get("brand", ""),
                }
            results.append({
                "product_id":     pid,
                "hybrid_score":   round(score, 4),
                **info,
            })
        return results

    # ── Explanation ────────────────────────────────────────────────────────────
    def explain(self, user_id: int, product_id: int,
                rated_products=None, rated_scores=None) -> dict:
        explanation = {}
        if self.cf and self.cf._trained:
            r = self.cf.predict_rating(user_id, product_id)
            explanation["collaborative_filter"] = round(r, 2)
        if self.cbf and rated_products:
            similar = self.cbf.similar_products(product_id, n=3)
            explanation["content_based"] = [s["product_name"] for s in similar]
        if self.ncf and self.ncf._trained:
            r = self.ncf.predict_rating(user_id, product_id)
            explanation["neural_cf"] = round(r, 2)
        return explanation

    # ── Persistence ────────────────────────────────────────────────────────────
    def save(self, path: str = "models/hybrid.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"💾 Hybrid model saved → {path}")

    @staticmethod
    def load(path: str = "models/hybrid.pkl") -> "HybridRecommender":
        with open(path, "rb") as f:
            return pickle.load(f)
