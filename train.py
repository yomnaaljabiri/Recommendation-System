"""
Training Pipeline — Flat Structure
يشتغل مع الهيكل:
  recommendation/
  ├── train.py
  ├── preprocessor.py         (أو data/preprocessor.py)
  ├── collaborative_filtering.py
  ├── content_based.py
  ├── hybrid.py
  ├── users.csv
  ├── products.csv
  ├── ratings.csv
  ├── purchases.csv
  └── models/
"""
import sys, os

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import pandas as pd
import pickle, json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse

MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.join(ROOT, "evaluation"), exist_ok=True)

print("\n" + "═"*60)
print("  SMART RECOMMENDATION SYSTEM — TRAINING PIPELINE")
print("═"*60 + "\n")

# ────────────────────────────────────────────────
# STEP 1: Load CSV files
# ────────────────────────────────────────────────
print("📦 STEP 1: Loading CSV files")

# يبحث عن CSV في نفس المجلد أو في مجلد data/
def find_csv(name):
    paths = [
        os.path.join(ROOT, name),
        os.path.join(ROOT, "data", name),
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"❌ ما لقيت {name} — تأكد أنه موجود في {ROOT}")

users     = pd.read_csv(find_csv("users.csv"))
products  = pd.read_csv(find_csv("products.csv"))
ratings   = pd.read_csv(find_csv("ratings.csv"))
purchases = pd.read_csv(find_csv("purchases.csv"))
print(f"  ✅ users={len(users)} | products={len(products)} | ratings={len(ratings)}")

# ────────────────────────────────────────────────
# STEP 2: Preprocessing
# ────────────────────────────────────────────────
print("\n🔧 STEP 2: Preprocessing")

user_enc    = LabelEncoder().fit(users["user_id"])
product_enc = LabelEncoder().fit(products["product_id"])

ratings["user_idx"]    = user_enc.transform(ratings["user_id"])
ratings["product_idx"] = product_enc.transform(ratings["product_id"])

n_users  = len(user_enc.classes_)
n_items  = len(product_enc.classes_)

# User-Item matrix
ui_matrix = scipy.sparse.csr_matrix(
    (ratings["rating"], (ratings["user_idx"], ratings["product_idx"])),
    shape=(n_users, n_items)
)
ui_dense = ui_matrix.toarray()

# Train/test split
train_df, test_df = train_test_split(ratings, test_size=0.2, random_state=42)

# Save ratings_encoded
ratings.to_csv(os.path.join(ROOT, "ratings_encoded.csv"), index=False)
scipy.sparse.save_npz(os.path.join(MODEL_DIR, "user_item_matrix.npz"), ui_matrix)
print(f"  ✅ Matrix shape: {ui_dense.shape} | Train: {len(train_df)} | Test: {len(test_df)}")

# ────────────────────────────────────────────────
# STEP 3: SVD
# ────────────────────────────────────────────────
print("\n🔬 STEP 3: SVD Collaborative Filtering")

svd_model      = TruncatedSVD(n_components=50, random_state=42)
U              = svd_model.fit_transform(ui_dense)
VT             = svd_model.components_
reconstructed  = U @ VT
print(f"  ✅ Explained variance: {svd_model.explained_variance_ratio_.sum():.2%}")

class SVDRec:
    def __init__(self, matrix, reconstructed):
        self.matrix        = matrix
        self.reconstructed = reconstructed
    def recommend(self, user_idx, top_k=10, exclude_seen=True):
        scores = self.reconstructed[user_idx].copy()
        if exclude_seen:
            scores[self.matrix[user_idx] > 0] = -np.inf
        idx = np.argsort(scores)[::-1][:top_k]
        return idx, scores[idx]

svd_rec = SVDRec(ui_dense, reconstructed)
with open(os.path.join(MODEL_DIR, "svd_model.pkl"), "wb") as f:
    pickle.dump(svd_rec, f)
print("  ✅ SVD saved")

# ────────────────────────────────────────────────
# STEP 4: Item-Item CF
# ────────────────────────────────────────────────
print("\n🧠 STEP 4: Item-Item CF")

item_sim = cosine_similarity(ui_dense.T)

class ItemCF:
    def __init__(self, matrix, similarity):
        self.matrix     = matrix
        self.similarity = similarity
    def recommend(self, user_idx, top_k=10, exclude_seen=True):
        scores = self.matrix[user_idx] @ self.similarity
        if exclude_seen:
            scores = scores.copy()
            scores[self.matrix[user_idx] > 0] = -np.inf
        idx = np.argsort(scores)[::-1][:top_k]
        return idx, scores[idx]

item_cf = ItemCF(ui_dense, item_sim)
with open(os.path.join(MODEL_DIR, "item_cf_model.pkl"), "wb") as f:
    pickle.dump(item_cf, f)
print("  ✅ Item-CF saved")

# ────────────────────────────────────────────────
# STEP 5: Content-Based
# ────────────────────────────────────────────────
print("\n📄 STEP 5: Content-Based Filtering")

products["content_text"] = (
    products["category"].str.lower() + " " +
    products["brand"].str.lower() + " " +
    products["tags"].str.replace("|", " ", regex=False)
)
tfidf        = TfidfVectorizer(max_features=150, stop_words="english")
tfidf_matrix = tfidf.fit_transform(products["content_text"])
item_sim_cb  = cosine_similarity(tfidf_matrix)
product_ids_list = products["product_id"].tolist()

class ContentBased:
    def __init__(self, products, tfidf_matrix, item_sim, product_ids):
        self.products     = products
        self.tfidf_matrix = tfidf_matrix
        self.item_sim     = item_sim
        self.product_ids  = product_ids

    def get_similar_items(self, product_id, top_k=10):
        idx    = self.product_ids.index(product_id)
        scores = self.item_sim[idx].copy()
        scores[idx] = -1
        top = np.argsort(scores)[::-1][:top_k]
        return top, scores[top]

    def recommend(self, user_id, user_ratings_df, top_k=10, exclude_seen=True):
        ur      = user_ratings_df[user_ratings_df["user_id"] == user_id]
        profile = np.zeros(self.tfidf_matrix.shape[1])
        w_sum   = 0
        for _, row in ur.iterrows():
            pid = row["product_id"]
            if pid not in self.product_ids: continue
            w = max(0, row["rating"] - 2.5)
            profile += w * self.tfidf_matrix[self.product_ids.index(pid)].toarray().flatten()
            w_sum += w
        if w_sum > 0: profile /= w_sum
        if profile.sum() == 0:
            top = self.products.nlargest(top_k, "avg_rating").index.tolist()
            return np.array(top), np.zeros(top_k)
        scores = cosine_similarity(profile.reshape(1,-1), self.tfidf_matrix)[0]
        if exclude_seen:
            seen = set(ur["product_id"])
            for pid in seen:
                if pid in self.product_ids:
                    scores[self.product_ids.index(pid)] = -np.inf
        top = np.argsort(scores)[::-1][:top_k]
        return top, scores[top]

    def recommend_by_category(self, category, top_k=10):
        df  = self.products[self.products["category"].str.lower() == category.lower()]
        df  = df.sort_values("avg_rating", ascending=False)
        idx = df.index[:top_k].tolist()
        return np.array(idx), df["avg_rating"].values[:top_k]

cb_model = ContentBased(products, tfidf_matrix, item_sim_cb, product_ids_list)
with open(os.path.join(MODEL_DIR, "content_based_model.pkl"), "wb") as f:
    pickle.dump(cb_model, f)
print("  ✅ Content-Based saved")

# ────────────────────────────────────────────────
# STEP 6: Hybrid
# ────────────────────────────────────────────────
print("\n🔗 STEP 6: Hybrid Recommender")

def normalize(scores):
    finite = scores[np.isfinite(scores)]
    if len(finite) == 0 or finite.max() == finite.min():
        return np.zeros_like(scores)
    s = scores.copy()
    mn, mx = finite.min(), finite.max()
    s[np.isfinite(s)] = (s[np.isfinite(s)] - mn) / (mx - mn)
    s[~np.isfinite(s)] = 0.0
    return s

class Hybrid:
    def __init__(self, svd, cb, ui_dense, products, ratings_df, cf_w=0.6, cb_w=0.4):
        self.svd       = svd
        self.cb        = cb
        self.ui_dense  = ui_dense
        self.products  = products
        self.ratings   = ratings_df
        self.cf_w      = cf_w
        self.cb_w      = cb_w

    def recommend(self, user_idx, user_id, top_k=10, exclude_seen=True):
        cf_scores = self.svd.reconstructed[user_idx].copy()
        profile   = self.cb.recommend.__func__(self.cb, user_id, self.ratings,
                                               top_k=len(self.cb.product_ids),
                                               exclude_seen=False)
        # Get CB scores properly
        ur      = self.ratings[self.ratings["user_id"] == user_id]
        profile_vec = np.zeros(self.cb.tfidf_matrix.shape[1])
        w_sum = 0
        for _, row in ur.iterrows():
            pid = row["product_id"]
            if pid not in self.cb.product_ids: continue
            w = max(0, row["rating"] - 2.5)
            profile_vec += w * self.cb.tfidf_matrix[self.cb.product_ids.index(pid)].toarray().flatten()
            w_sum += w
        if w_sum > 0: profile_vec /= w_sum
        if profile_vec.sum() > 0:
            cb_scores = cosine_similarity(profile_vec.reshape(1,-1), self.cb.tfidf_matrix)[0]
        else:
            cb_scores = np.zeros(len(self.cb.product_ids))

        fused = self.cf_w * normalize(cf_scores) + self.cb_w * normalize(cb_scores)
        if exclude_seen:
            fused[self.ui_dense[user_idx] > 0] = -np.inf
        top = np.argsort(fused)[::-1][:top_k]
        return top, fused[top]

hybrid = Hybrid(svd_rec, cb_model, ui_dense, products, ratings)
with open(os.path.join(MODEL_DIR, "hybrid_model.pkl"), "wb") as f:
    pickle.dump(hybrid, f)
print("  ✅ Hybrid saved")

# ────────────────────────────────────────────────
# STEP 7: Evaluation
# ────────────────────────────────────────────────
print("\n📊 STEP 7: Evaluation")

def precision_at_k(rec, relevant, k):
    return sum(1 for i in rec[:k] if i in relevant) / k if k else 0

def recall_at_k(rec, relevant, k):
    return sum(1 for i in rec[:k] if i in relevant) / len(relevant) if relevant else 0

p_scores, r_scores = [], []
test_users = test_df["user_idx"].unique()[:200]
for uid in test_users:
    ut       = test_df[test_df["user_idx"] == uid]
    relevant = set(ut[ut["rating"] >= 3.5]["product_idx"].tolist())
    if not relevant: continue
    top, _   = svd_rec.recommend(uid, top_k=10)
    p_scores.append(precision_at_k(top.tolist(), relevant, 10))
    r_scores.append(recall_at_k(top.tolist(), relevant, 10))

# RMSE
y_true, y_pred = [], []
for _, row in test_df.iterrows():
    y_true.append(row["rating"])
    y_pred.append(float(reconstructed[int(row["user_idx"]), int(row["product_idx"])]))

rmse = float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2)))
mae  = float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))

report = {
    "ranking": {
        "@10": {
            "precision": round(np.mean(p_scores), 4),
            "recall":    round(np.mean(r_scores), 4),
        }
    },
    "regression": {"rmse": round(rmse,4), "mae": round(mae,4)}
}
with open(os.path.join(ROOT, "evaluation", "eval_report.json"), "w") as f:
    json.dump(report, f, indent=2)

print(f"  Precision@10 : {np.mean(p_scores):.4f}")
print(f"  Recall@10    : {np.mean(r_scores):.4f}")
print(f"  RMSE         : {rmse:.4f}")
print(f"  MAE          : {mae:.4f}")

# ────────────────────────────────────────────────
# STEP 8: Save metadata for API
# ────────────────────────────────────────────────
print("\n💾 STEP 8: Saving metadata")

# Save preprocessor-like object
class PrepMeta:
    pass
prep = PrepMeta()
prep.user_encoder    = user_enc
prep.product_encoder = product_enc
prep.n_users         = n_users
prep.n_products      = n_items
prep.train_ratings   = train_df
prep.test_ratings    = test_df

with open(os.path.join(MODEL_DIR, "preprocessor.pkl"), "wb") as f:
    pickle.dump(prep, f)

meta = {
    "n_users":     n_users,
    "n_items":     n_items,
    "user_ids":    user_enc.classes_.tolist(),
    "product_ids": product_enc.classes_.tolist(),
    "categories":  products["category"].unique().tolist(),
}
with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("\n✅ Pipeline complete! Models saved to /models/")
print(f"   RMSE: {rmse:.4f} | Precision@10: {np.mean(p_scores):.4f}")
