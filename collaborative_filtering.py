"""
Collaborative Filtering Models  —  Pure NumPy/Scikit-learn (NO PyTorch)
1. Memory-Based : Item-Item & User-User cosine similarity
2. SVD          : TruncatedSVD matrix factorization
3. NeuralCF     : Embedding + MLP implemented in pure NumPy
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import pickle, os

MODEL_DIR = os.path.dirname(__file__)
DATA_DIR  = os.path.join(os.path.dirname(__file__), "../data")


# ═══════════════════════════════════════════════════
# 1. Memory-Based Collaborative Filtering
# ═══════════════════════════════════════════════════
class MemoryBasedCF:
    def __init__(self, mode="item"):
        assert mode in ("user", "item")
        self.mode = mode

    def fit(self, user_item_matrix: np.ndarray):
        self.matrix = user_item_matrix
        if self.mode == "item":
            self.similarity = cosine_similarity(user_item_matrix.T)
        else:
            self.similarity = cosine_similarity(user_item_matrix)
        print(f"[MemoryCF-{self.mode}] similarity matrix: {self.similarity.shape}")
        return self

    def recommend(self, user_idx: int, top_k: int = 10, exclude_seen: bool = True):
        user_ratings = self.matrix[user_idx]
        if self.mode == "item":
            scores = user_ratings @ self.similarity
        else:
            scores = self.similarity[user_idx] @ self.matrix
        if exclude_seen:
            scores = scores.copy()
            scores[user_ratings > 0] = -np.inf
        top_idx = np.argsort(scores)[::-1][:top_k]
        return top_idx, scores[top_idx]


# ═══════════════════════════════════════════════════
# 2. SVD Matrix Factorization
# ═══════════════════════════════════════════════════
class SVDRecommender:
    def __init__(self, n_components: int = 50):
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)

    def fit(self, user_item_matrix: np.ndarray):
        self.matrix        = user_item_matrix
        self.U             = self.svd.fit_transform(user_item_matrix)
        self.VT            = self.svd.components_
        self.reconstructed = self.U @ self.VT
        print(f"[SVD] Explained variance: {self.svd.explained_variance_ratio_.sum():.2%}")
        return self

    def recommend(self, user_idx: int, top_k: int = 10, exclude_seen: bool = True):
        scores = self.reconstructed[user_idx].copy()
        if exclude_seen:
            scores[self.matrix[user_idx] > 0] = -np.inf
        top_idx = np.argsort(scores)[::-1][:top_k]
        return top_idx, scores[top_idx]

    def get_user_embedding(self, user_idx): return self.U[user_idx]
    def get_item_embedding(self, item_idx): return self.VT[:, item_idx]


# ═══════════════════════════════════════════════════
# 3. Neural CF — Pure NumPy  (GMF + MLP fusion)
# ═══════════════════════════════════════════════════
def _sigmoid(x):  return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
def _relu(x):     return np.maximum(0, x)
def _relu_grad(x):return (x > 0).astype(np.float32)


class NeuralCFNumpy:
    """
    Neural Matrix Factorization — pure NumPy.
    GMF path  : embed_user  ⊙  embed_item
    MLP path  : [embed_user ‖ embed_item] → ReLU layers
    Output    : sigmoid([GMF ‖ MLP]) scaled to [1, 5]
    """

    def __init__(self, n_users, n_items,
                 embedding_dim=16, mlp_hidden=None,
                 lr=0.005, epochs=20, batch_size=512, reg=1e-4):
        if mlp_hidden is None:
            mlp_hidden = [64, 32]
        self.n_users       = n_users
        self.n_items       = n_items
        self.embedding_dim = embedding_dim
        self.mlp_hidden    = mlp_hidden
        self.lr            = lr
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.reg           = reg
        self.history       = {"train_loss": [], "val_loss": []}
        self._init_weights()

    def _init_weights(self):
        d, rng, s = self.embedding_dim, np.random.default_rng(42), 0.01
        self.E_gmf_u = rng.normal(0, s, (self.n_users, d)).astype(np.float32)
        self.E_gmf_i = rng.normal(0, s, (self.n_items, d)).astype(np.float32)
        self.E_mlp_u = rng.normal(0, s, (self.n_users, d)).astype(np.float32)
        self.E_mlp_i = rng.normal(0, s, (self.n_items, d)).astype(np.float32)
        self.mlp_layers = []
        in_dim = d * 2
        for out_dim in self.mlp_hidden:
            W = rng.normal(0, np.sqrt(2.0/in_dim), (in_dim, out_dim)).astype(np.float32)
            b = np.zeros(out_dim, dtype=np.float32)
            self.mlp_layers.append([W, b])
            in_dim = out_dim
        out_in = d + self.mlp_hidden[-1]
        self.W_out = rng.normal(0, s, (out_in, 1)).astype(np.float32)
        self.b_out = np.zeros(1, dtype=np.float32)

    def _forward(self, user_ids, item_ids):
        eu_gmf = self.E_gmf_u[user_ids]
        ei_gmf = self.E_gmf_i[item_ids]
        eu_mlp = self.E_mlp_u[user_ids]
        ei_mlp = self.E_mlp_i[item_ids]
        gmf = eu_gmf * ei_gmf
        h   = np.concatenate([eu_mlp, ei_mlp], axis=1)
        activations, pre_acts = [h], []
        for W, b in self.mlp_layers:
            z = h @ W + b;  pre_acts.append(z);  h = _relu(z);  activations.append(h)
        fused = np.concatenate([gmf, h], axis=1)
        logit = fused @ self.W_out + self.b_out
        pred  = _sigmoid(logit).squeeze(1) * 4.0 + 1.0
        return pred, (user_ids, item_ids, eu_gmf, ei_gmf, eu_mlp, ei_mlp,
                      gmf, activations, pre_acts, h, fused, logit, pred)

    def _backward(self, cache, ratings):
        (user_ids, item_ids, eu_gmf, ei_gmf, eu_mlp, ei_mlp,
         gmf, activations, pre_acts, mlp_out, fused, logit, pred) = cache
        B, lr, reg = len(ratings), self.lr, self.reg
        d_pred  = 2.0 * (pred - ratings) / B
        sig     = _sigmoid(logit).squeeze(1)
        d_logit = (d_pred * 4.0 * sig * (1 - sig)).reshape(-1, 1)
        dW_out  = fused.T @ d_logit + reg * self.W_out
        db_out  = d_logit.sum(axis=0)
        d_fused = d_logit @ self.W_out.T
        d_gmf     = d_fused[:, :self.embedding_dim]
        d_mlp_out = d_fused[:, self.embedding_dim:]
        d_mlp_layer_grads = []
        dh = d_mlp_out
        for i in reversed(range(len(self.mlp_layers))):
            W, _ = self.mlp_layers[i]
            dz   = dh * _relu_grad(pre_acts[i])
            dW   = activations[i].T @ dz + reg * W
            dbl  = dz.sum(axis=0)
            dh   = dz @ W.T
            d_mlp_layer_grads.insert(0, (dW, dbl))
        d_eu_mlp = dh[:, :self.embedding_dim]
        d_ei_mlp = dh[:, self.embedding_dim:]
        self.W_out -= lr * dW_out
        self.b_out -= lr * db_out
        for i, (dW, dbl) in enumerate(d_mlp_layer_grads):
            self.mlp_layers[i][0] -= lr * dW
            self.mlp_layers[i][1] -= lr * dbl
        np.add.at(self.E_gmf_u, user_ids, -lr * (d_gmf * ei_gmf        + reg * eu_gmf))
        np.add.at(self.E_gmf_i, item_ids, -lr * (d_gmf * eu_gmf        + reg * ei_gmf))
        np.add.at(self.E_mlp_u, user_ids, -lr * (d_eu_mlp              + reg * eu_mlp))
        np.add.at(self.E_mlp_i, item_ids, -lr * (d_ei_mlp              + reg * ei_mlp))

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        tu = train_df["user_idx"].values.astype(np.int32)
        ti = train_df["product_idx"].values.astype(np.int32)
        tr = train_df["rating"].values.astype(np.float32)
        vu = val_df["user_idx"].values.astype(np.int32)
        vi = val_df["product_idx"].values.astype(np.int32)
        vr = val_df["rating"].values.astype(np.float32)
        n_params = (self.n_users + self.n_items) * self.embedding_dim * 2
        print(f"[NeuralCF-NumPy] ~{n_params:,} embedding params | CPU only")
        for epoch in range(self.epochs):
            perm = np.random.permutation(len(tu))
            tu, ti, tr = tu[perm], ti[perm], tr[perm]
            total, nb = 0.0, 0
            for s in range(0, len(tu), self.batch_size):
                bu, bi, br = tu[s:s+self.batch_size], ti[s:s+self.batch_size], tr[s:s+self.batch_size]
                pred, cache = self._forward(bu, bi)
                total += float(np.mean((pred - br)**2))
                self._backward(cache, br);  nb += 1
            vp, _ = self._forward(vu, vi)
            vl    = float(np.mean((vp - vr)**2))
            self.history["train_loss"].append(total/nb)
            self.history["val_loss"].append(vl)
            if (epoch+1) % 5 == 0:
                print(f"  Epoch {epoch+1:3d}/{self.epochs} | Train RMSE: {(total/nb)**0.5:.4f} | Val RMSE: {vl**0.5:.4f}")
            if (epoch+1) % 8 == 0:
                self.lr *= 0.7
        return self

    def predict_all_items(self, user_idx: int) -> np.ndarray:
        uids = np.full(self.n_items, user_idx, dtype=np.int32)
        iids = np.arange(self.n_items, dtype=np.int32)
        pred, _ = self._forward(uids, iids)
        return pred

    def recommend(self, user_idx: int, top_k: int = 10, exclude_seen_matrix=None):
        scores = self.predict_all_items(user_idx)
        if exclude_seen_matrix is not None:
            scores = scores.copy()
            scores[exclude_seen_matrix[user_idx] > 0] = -np.inf
        top_idx = np.argsort(scores)[::-1][:top_k]
        return top_idx, scores[top_idx]

    def save(self, path: str):
        with open(path, "wb") as f: pickle.dump(self, f)
        print(f"[NeuralCF-NumPy] Saved → {path}")

    @staticmethod
    def load(path: str):
        with open(path, "rb") as f: return pickle.load(f)


# Alias — train.py يستخدم هذا الاسم
NeuralCFTrainer = NeuralCFNumpy


if __name__ == "__main__":
    print("✅ collaborative_filtering.py (pure NumPy — no PyTorch) loaded")
