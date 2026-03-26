from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import numpy as np, pandas as pd, pickle, json, os
from sklearn.metrics.pairwise import cosine_similarity

ROOT=os.path.dirname(os.path.abspath(__file__))
MODEL_DIR=os.path.join(ROOT,"models")
EVAL_DIR=os.path.join(ROOT,"evaluation")

def find_file(name):
    for p in [os.path.join(ROOT,name),os.path.join(ROOT,"data",name)]:
        if os.path.exists(p): return p

class PrepMeta: pass

class SVDRec:
    def __init__(self,matrix,reconstructed): self.matrix=matrix; self.reconstructed=reconstructed
    def recommend(self,user_idx,top_k=10,exclude_seen=True):
        scores=self.reconstructed[user_idx].copy()
        if exclude_seen: scores[self.matrix[user_idx]>0]=-np.inf
        idx=np.argsort(scores)[::-1][:top_k]; return idx,scores[idx]

class ItemCF:
    def __init__(self,matrix,similarity): self.matrix=matrix; self.similarity=similarity
    def recommend(self,user_idx,top_k=10,exclude_seen=True):
        scores=self.matrix[user_idx]@self.similarity
        if exclude_seen: scores=scores.copy(); scores[self.matrix[user_idx]>0]=-np.inf
        idx=np.argsort(scores)[::-1][:top_k]; return idx,scores[idx]

class ContentBased:
    def __init__(self,products,tfidf_matrix,item_sim,product_ids):
        self.products=products; self.tfidf_matrix=tfidf_matrix
        self.item_sim=item_sim; self.product_ids=product_ids
    def get_similar_items(self,product_id,top_k=10):
        idx=self.product_ids.index(product_id); scores=self.item_sim[idx].copy(); scores[idx]=-1
        top=np.argsort(scores)[::-1][:top_k]; return top,scores[top]
    def recommend(self,user_id,user_ratings_df,top_k=10,exclude_seen=True):
        ur=user_ratings_df[user_ratings_df["user_id"]==user_id]
        profile=np.zeros(self.tfidf_matrix.shape[1]); w_sum=0
        for _,row in ur.iterrows():
            pid=row["product_id"]
            if pid not in self.product_ids: continue
            w=max(0,row["rating"]-2.5)
            profile+=w*self.tfidf_matrix[self.product_ids.index(pid)].toarray().flatten(); w_sum+=w
        if w_sum>0: profile/=w_sum
        if profile.sum()==0:
            top=self.products.nlargest(top_k,"avg_rating").index.tolist(); return np.array(top),np.zeros(top_k)
        scores=cosine_similarity(profile.reshape(1,-1),self.tfidf_matrix)[0]
        if exclude_seen:
            for pid in set(ur["product_id"]):
                if pid in self.product_ids: scores[self.product_ids.index(pid)]=-np.inf
        top=np.argsort(scores)[::-1][:top_k]; return top,scores[top]
    def recommend_by_category(self,category,top_k=10):
        df=self.products[self.products["category"].str.lower()==category.lower()].sort_values("avg_rating",ascending=False)
        return np.array(df.index[:top_k].tolist()),df["avg_rating"].values[:top_k]

class Hybrid:
    def __init__(self,svd,cb,ui_dense,products,ratings_df,cf_w=0.6,cb_w=0.4):
        self.svd=svd; self.cb=cb; self.ui_dense=ui_dense
        self.products=products; self.ratings=ratings_df; self.cf_w=cf_w; self.cb_w=cb_w
    def _norm(self,s):
        f=s[np.isfinite(s)]
        if len(f)==0 or f.max()==f.min(): return np.zeros_like(s)
        out=s.copy(); mn,mx=f.min(),f.max()
        out[np.isfinite(out)]=(out[np.isfinite(out)]-mn)/(mx-mn); out[~np.isfinite(out)]=0.0; return out
    def recommend(self,user_idx,user_id,top_k=10,exclude_seen=True):
        cf=self.svd.reconstructed[user_idx].copy()
        ur=self.ratings[self.ratings["user_id"]==user_id]
        profile=np.zeros(self.cb.tfidf_matrix.shape[1]); w=0
        for _,row in ur.iterrows():
            pid=row["product_id"]
            if pid not in self.cb.product_ids: continue
            wt=max(0,row["rating"]-2.5)
            profile+=wt*self.cb.tfidf_matrix[self.cb.product_ids.index(pid)].toarray().flatten(); w+=wt
        if w>0: profile/=w
        cb=cosine_similarity(profile.reshape(1,-1),self.cb.tfidf_matrix)[0] if profile.sum()>0 else np.zeros(len(self.cb.product_ids))
        fused=self.cf_w*self._norm(cf)+self.cb_w*self._norm(cb)
        if exclude_seen: fused[self.ui_dense[user_idx]>0]=-np.inf
        top=np.argsort(fused)[::-1][:top_k]; return top,fused[top]

app=FastAPI(title="Smart Recommendation System API",version="1.0.0")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])
prep=svd=item_cf=cb_model=hybrid=products_df=ratings_df=meta=None

@app.on_event("startup")
def load_models():
    global prep,svd,item_cf,cb_model,hybrid,products_df,ratings_df,meta
    try:
        with open(f"{MODEL_DIR}/preprocessor.pkl","rb") as f: prep=pickle.load(f)
        with open(f"{MODEL_DIR}/svd_model.pkl","rb") as f: svd=pickle.load(f)
        with open(f"{MODEL_DIR}/item_cf_model.pkl","rb") as f: item_cf=pickle.load(f)
        with open(f"{MODEL_DIR}/content_based_model.pkl","rb") as f: cb_model=pickle.load(f)
        with open(f"{MODEL_DIR}/hybrid_model.pkl","rb") as f: hybrid=pickle.load(f)
        with open(f"{MODEL_DIR}/meta.json") as f: meta=json.load(f)
        products_df=pd.read_csv(find_file("products.csv"))
        ratings_df=pd.read_csv(find_file("ratings_encoded.csv"))
        print("✅ All models loaded")
    except Exception as e: print(f"⚠️ {e}"); raise

def idx_to_product(product_idx,score,reason=""):
    pid=prep.product_encoder.inverse_transform([int(product_idx)])[0]
    row=products_df[products_df["product_id"]==pid]
    if row.empty: return None
    r=row.iloc[0]
    return {"product_id":pid,"name":r["name"],"category":r["category"],"brand":r["brand"],"score":round(float(score),4),"reason":reason}

@app.get("/health")
def health(): return {"status":"healthy","models_loaded":prep is not None}

@app.get("/users")
def list_users(limit:int=Query(200,le=500)): return {"users":meta["user_ids"][:limit],"total":len(meta["user_ids"])}

@app.get("/products")
def list_products(category:Optional[str]=None,limit:int=Query(20,le=200)):
    df=products_df.copy()
    if category: df=df[df["category"].str.lower()==category.lower()]
    return {"products":df.head(limit).to_dict(orient="records"),"total":len(df)}

@app.get("/recommend/{user_id}")
def recommend(user_id:str,method:str=Query("hybrid",enum=["hybrid","svd","item_cf","content"]),top_k:int=Query(10,ge=1,le=50)):
    if user_id not in meta["user_ids"]: raise HTTPException(404,f"User {user_id} not found")
    user_idx=int(prep.user_encoder.transform([user_id])[0]); recs=[]
    try:
        if method=="svd":
            for i,s in zip(*svd.recommend(user_idx,top_k=top_k)):
                p=idx_to_product(i,s,"SVD"); recs.append(p) if p else None
        elif method=="item_cf":
            for i,s in zip(*item_cf.recommend(user_idx,top_k=top_k)):
                p=idx_to_product(i,s,"Item-CF"); recs.append(p) if p else None
        elif method=="content":
            for i,s in zip(*cb_model.recommend(user_id,ratings_df,top_k=top_k)):
                p=idx_to_product(i,s,"Content"); recs.append(p) if p else None
        else:
            for i,s in zip(*hybrid.recommend(user_idx,user_id,top_k=top_k)):
                p=idx_to_product(i,s,"Hybrid"); recs.append(p) if p else None
    except Exception as e: raise HTTPException(500,str(e))
    return {"user_id":user_id,"method":method,"recommendations":recs,"total":len(recs)}

@app.get("/similar/{product_id}")
def similar(product_id:str,top_k:int=Query(10,ge=1,le=50)):
    if product_id not in meta["product_ids"]: raise HTTPException(404,"Not found")
    indices,scores=cb_model.get_similar_items(product_id,top_k=top_k); result=[]
    for idx,sc in zip(indices,scores):
        pid=cb_model.product_ids[idx]; row=products_df[products_df["product_id"]==pid]
        if not row.empty:
            r=row.iloc[0]; result.append({"product_id":pid,"name":r["name"],"category":r["category"],"brand":r["brand"],"similarity_score":round(float(sc),4)})
    return {"product_id":product_id,"similar_products":result}

@app.get("/popular")
def popular(top_k:int=Query(10,ge=1,le=50),category:Optional[str]=None):
    df=products_df.copy()
    if category: df=df[df["category"].str.lower()==category.lower()]
    return {"popular":df.nlargest(top_k,"avg_rating").to_dict(orient="records")}

@app.get("/category/{category}")
def by_category(category:str,top_k:int=Query(10,ge=1,le=50)):
    indices,scores=cb_model.recommend_by_category(category,top_k=top_k); result=[]
    for idx,sc in zip(indices,scores):
        pid=cb_model.product_ids[int(idx)]; row=products_df[products_df["product_id"]==pid]
        if not row.empty:
            r=row.iloc[0]; result.append({"product_id":pid,"name":r["name"],"category":r["category"],"brand":r["brand"],"avg_rating":float(r.get("avg_rating",sc))})
    return {"category":category,"recommendations":result}

@app.get("/metrics")
def get_metrics():
    path=os.path.join(EVAL_DIR,"eval_report.json")
    if not os.path.exists(path): return {"error":"Run train.py first"}
    with open(path) as f: return json.load(f)

if __name__=="__main__":
    import uvicorn; uvicorn.run("main:app",host="0.0.0.0",port=8000,reload=True)
