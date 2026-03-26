"""
models/neural_cf.py
Neural Collaborative Filtering with Embedding layers (PyTorch).
Architecture: GMF + MLP → NeuMF (He et al., 2017)
"""
import os, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# ── Dataset ───────────────────────────────────────────────────────────────────
class RatingDataset(Dataset):
    def __init__(self, users, products, ratings):
        self.users    = torch.LongTensor(users)
        self.products = torch.LongTensor(products)
        self.ratings  = torch.FloatTensor(ratings)

    def __len__(self):  return len(self.ratings)
    def __getitem__(self, idx):
        return self.users[idx], self.products[idx], self.ratings[idx]


# ── NeuMF Model ───────────────────────────────────────────────────────────────
class NeuMF(nn.Module):
    """
    Neural Matrix Factorisation:
    - GMF path  : element-wise product of user/item embeddings
    - MLP path  : concatenation → deep layers
    - Output    : sigmoid → scale to [1, 5]
    """

    def __init__(self, n_users, n_items,
                 embed_dim=32, mlp_layers=(64, 32, 16), dropout=0.2):
        super().__init__()

        # GMF embeddings
        self.gmf_user = nn.Embedding(n_users + 1, embed_dim)
        self.gmf_item = nn.Embedding(n_items + 1, embed_dim)

        # MLP embeddings
        self.mlp_user = nn.Embedding(n_users + 1, embed_dim)
        self.mlp_item = nn.Embedding(n_items + 1, embed_dim)

        # MLP tower
        mlp_input = embed_dim * 2
        layers    = []
        for out_dim in mlp_layers:
            layers += [nn.Linear(mlp_input, out_dim), nn.ReLU(),
                       nn.Dropout(dropout)]
            mlp_input = out_dim
        self.mlp = nn.Sequential(*layers)

        # Output head
        self.output = nn.Linear(embed_dim + mlp_layers[-1], 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, user, item):
        # GMF
        gmf_out = self.gmf_user(user) * self.gmf_item(item)
        # MLP
        mlp_in  = torch.cat([self.mlp_user(user), self.mlp_item(item)], dim=1)
        mlp_out = self.mlp(mlp_in)
        # Concat & predict
        x   = torch.cat([gmf_out, mlp_out], dim=1)
        out = self.output(x).squeeze()
        return torch.sigmoid(out) * 4 + 1          # scale → [1, 5]


# ── Trainer / Wrapper ─────────────────────────────────────────────────────────
class NeuralCF:

    def __init__(self, embed_dim=32, mlp_layers=(64, 32, 16),
                 lr=1e-3, batch_size=256, epochs=15, dropout=0.2):
        self.embed_dim   = embed_dim
        self.mlp_layers  = mlp_layers
        self.lr          = lr
        self.batch_size  = batch_size
        self.epochs      = epochs
        self.dropout     = dropout
        self.model       = None
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.user_map    = {}   # raw_id → 0-indexed
        self.item_map    = {}
        self.all_item_ids = []
        self._trained    = False

    # ── Training ──────────────────────────────────────────────────────────────
    def train(self, ratings_df: pd.DataFrame):
        # Build index maps
        users    = ratings_df["user_id"].unique()
        items    = ratings_df["product_id"].unique()
        self.user_map  = {u: i+1 for i, u in enumerate(users)}
        self.item_map  = {p: i+1 for i, p in enumerate(items)}
        self.all_item_ids = list(items)

        df = ratings_df.copy()
        df["u_idx"] = df["user_id"].map(self.user_map)
        df["i_idx"] = df["product_id"].map(self.item_map)

        train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)

        train_ds = RatingDataset(train_df["u_idx"].values,
                                 train_df["i_idx"].values,
                                 train_df["rating"].values.astype(np.float32))
        val_ds   = RatingDataset(val_df["u_idx"].values,
                                 val_df["i_idx"].values,
                                 val_df["rating"].values.astype(np.float32))

        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_dl   = DataLoader(val_ds,   batch_size=self.batch_size)

        n_users = len(self.user_map)
        n_items = len(self.item_map)
        self.model = NeuMF(n_users, n_items, self.embed_dim,
                           self.mlp_layers, self.dropout).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                     weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.5)
        criterion = nn.MSELoss()

        best_val, best_state = float("inf"), None

        for epoch in range(1, self.epochs + 1):
            # Train
            self.model.train()
            t_loss = 0.0
            for u, i, r in train_dl:
                u, i, r = u.to(self.device), i.to(self.device), r.to(self.device)
                optimizer.zero_grad()
                pred = self.model(u, i)
                loss = criterion(pred, r)
                loss.backward()
                optimizer.step()
                t_loss += loss.item()

            # Validate
            self.model.eval()
            v_loss = 0.0
            with torch.no_grad():
                for u, i, r in val_dl:
                    u, i, r = u.to(self.device), i.to(self.device), r.to(self.device)
                    v_loss += criterion(self.model(u, i), r).item()

            t_rmse = (t_loss / len(train_dl)) ** 0.5
            v_rmse = (v_loss / len(val_dl))   ** 0.5
            scheduler.step(v_rmse)

            if v_rmse < best_val:
                best_val   = v_rmse
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{self.epochs} │ "
                      f"train RMSE {t_rmse:.4f} │ val RMSE {v_rmse:.4f}")

        self.model.load_state_dict(best_state)
        self._trained = True
        print(f"✅ NeuMF trained. Best val RMSE: {best_val:.4f}")
        return {"best_val_rmse": round(best_val, 4)}

    # ── Inference ─────────────────────────────────────────────────────────────
    def predict_rating(self, user_id: int, product_id: int) -> float:
        if not self._trained:
            raise RuntimeError("Train first.")
        u = self.user_map.get(user_id, 0)
        i = self.item_map.get(product_id, 0)
        self.model.eval()
        with torch.no_grad():
            u_t = torch.LongTensor([u]).to(self.device)
            i_t = torch.LongTensor([i]).to(self.device)
            return float(self.model(u_t, i_t).cpu())

    def recommend(self, user_id: int, n: int = 10,
                  rated_products: list = None) -> list[dict]:
        rated = set(rated_products or [])
        candidates = [p for p in self.all_item_ids if p not in rated]

        u_idx = self.user_map.get(user_id, 0)
        self.model.eval()
        results = []
        with torch.no_grad():
            # Batch predict
            chunk = 512
            all_scores = []
            for start in range(0, len(candidates), chunk):
                batch = candidates[start:start+chunk]
                u_t = torch.LongTensor([u_idx] * len(batch)).to(self.device)
                i_t = torch.LongTensor(
                    [self.item_map.get(p, 0) for p in batch]
                ).to(self.device)
                scores = self.model(u_t, i_t).cpu().numpy()
                all_scores.extend(zip(batch, scores))

        all_scores.sort(key=lambda x: x[1], reverse=True)
        for p, s in all_scores[:n]:
            results.append({"product_id": p, "predicted_rating": round(float(s), 2)})
        return results

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self, path: str = "models/ncf_model.pkl"):
        state = self.model.state_dict() if self.model else None
        payload = {
            "config":      {k: v for k, v in self.__dict__.items()
                            if k not in ("model", "device")},
            "state_dict":  state,
            "user_map":    self.user_map,
            "item_map":    self.item_map,
            "all_item_ids": self.all_item_ids,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print(f"💾 NeuMF saved → {path}")

    @staticmethod
    def load(path: str = "models/ncf_model.pkl") -> "NeuralCF":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        obj = NeuralCF(**{k: v for k, v in payload["config"].items()
                         if k not in ("_trained",)})
        obj.user_map    = payload["user_map"]
        obj.item_map    = payload["item_map"]
        obj.all_item_ids = payload["all_item_ids"]

        n_users = len(obj.user_map)
        n_items = len(obj.item_map)
        obj.model = NeuMF(n_users, n_items, obj.embed_dim,
                          obj.mlp_layers, obj.dropout).to(obj.device)
        obj.model.load_state_dict(payload["state_dict"])
        obj._trained = True
        return obj
