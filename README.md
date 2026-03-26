[README.md](https://github.com/user-attachments/files/26288693/README.md)
# 🧠 Smart Product Recommendation System

> An end-to-end AI-powered recommendation system combining Collaborative Filtering, Content-Based Filtering, and Neural CF — deployed with FastAPI and a real-time interactive dashboard.

[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?style=flat-square&logo=scikit-learn)](https://scikit-learn.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-blue?style=flat-square&logo=numpy)](https://numpy.org)

---

## 🔗 Live Demo

**[🌐 View Live Dashboard](https://yomnaaljabiri.github.io/Recommendation-System/)**

---

## 📌 Overview

A full-stack recommendation system that combines multiple ML/DL approaches into a hybrid model, served through a REST API and visualized in a real-time web dashboard — no frameworks, no cloud costs.

| Component | Technology |
|-----------|-----------|
| Matrix Factorization | TruncatedSVD (scikit-learn) |
| Item-Item CF | Cosine Similarity |
| Content-Based Filtering | TF-IDF + Cosine Similarity |
| Neural CF | GMF + MLP Embeddings (pure NumPy) |
| Hybrid Model | Weighted Score Fusion (60% CF + 40% CB) |
| REST API | FastAPI + Uvicorn |
| Frontend | Vanilla HTML / CSS / JavaScript |

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| RMSE | 1.7534 |
| MAE | 1.5660 |
| SVD Explained Variance | 48.18% |
| Training Samples | 6,400 |
| Test Samples | 1,600 |

---

## 🗂️ Dataset

Synthetic e-commerce dataset generated to simulate real user behavior.

| File | Size | Description |
|------|------|-------------|
| users.csv | 500 users | Age, location, segment |
| products.csv | 200 products | 8 categories, price, rating |
| ratings.csv | 8,000 ratings | Ratings from 1 to 5 |
| purchases.csv | 3,000 purchases | Purchase history |

**8 Product Categories:** Electronics · Books · Clothing · Sports · Home · Beauty · Food · Toys

---

## 🏗️ Project Structure

```
recommendation-system/
├── index.html                    # Interactive web dashboard
├── data.js                       # Embedded dataset (JS)
├── main.py                       # FastAPI application
├── train.py                      # Full training pipeline
├── collaborative_filtering.py    # SVD + Item-CF + Neural CF
├── content_based.py              # TF-IDF Content-Based Filter
├── hybrid.py                     # Hybrid Recommender
├── data/
│   ├── users.csv
│   ├── products.csv
│   ├── ratings.csv
│   └── purchases.csv
├── models/                       # Saved model artifacts
├── evaluation/                   # Evaluation reports
└── requirements.txt
```

---

## 🤖 Model Architecture

### 1. SVD Matrix Factorization
```
User-Item Matrix  (500 × 200)
        ↓  TruncatedSVD  (k = 50)
   U (500×50)  ·  Σ  ·  Vt (50×200)
        ↓
Reconstructed Matrix  →  Top-K Recommendations
```

### 2. Neural Collaborative Filtering (GMF + MLP)
```
                  ┌─ GMF Path ─┐
User Embedding ───┤             ├─── element-wise ⊙
Item Embedding ───┘             │
                                │
User Embedding ───┐             ├─── concat → [128→64→32] → Dense(1)
Item Embedding ───┴─ MLP Path ──┘
                        ↓
              sigmoid(x) × 4 + 1   →   predicted rating [1, 5]
```

### 3. Hybrid Score Fusion
```
final_score = 0.6 × normalize(CF_score)
            + 0.4 × normalize(Content_score)
```

---

## 🚀 Running Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train all models
python train.py

# 3. Start the API
python -m uvicorn main:app --port 8000

# 4. Open the dashboard
# Double-click index.html  (no server needed)
```

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | API health check |
| GET | `/users` | List all users |
| GET | `/products` | List all products |
| GET | `/recommend/{user_id}?method=hybrid` | Personalized recommendations |
| GET | `/similar/{product_id}` | Find similar products |
| GET | `/popular?category=Electronics` | Top rated products |
| GET | `/metrics` | Evaluation report |

**Recommendation methods:** `hybrid` · `svd` · `item_cf` · `content`

---

## 🛠️ Skills Demonstrated

- **Machine Learning** — SVD Matrix Factorization, TF-IDF, Cosine Similarity, scikit-learn pipelines
- **Deep Learning** — Embedding layers, GMF + MLP Neural CF implemented from scratch in NumPy
- **Data Engineering** — Pandas, Scipy sparse matrices, Label encoding, Train/test splitting
- **Model Evaluation** — RMSE, MAE, Precision@K, Recall@K, NDCG@K
- **API Development** — FastAPI, Uvicorn, REST design, CORS, Pydantic schemas
- **Frontend** — Vanilla JS, real-time filtering, pagination, data visualization

---

## 👨‍💻 Author

**Yomna aljabiri** — [https://www.linkedin.com/in/yomna-aljabiri/](#) · [https://github.com/yomnaaljabiri](#)
