# index/search_server.py
import json, os
import yaml
from typing import List

import faiss
import numpy as np
from fastapi import FastAPI, Query
from pydantic import BaseModel

# Optional rerank with trained head
import torch
from models.heads import ProjectionHead
import pandas as pd

# -------------------- Config & Data --------------------
cfg = yaml.safe_load(open("config.yaml"))

ids = json.load(open(cfg["paths"]["ids_json"]))
index = faiss.read_index(cfg["paths"]["faiss_index"])
metric = cfg["index"]["metric"]
normalize = (metric == "cosine")

X = np.load(cfg["paths"]["embeddings_npy"]).astype("float32")

# Optional metadata (for category filter)
ID2CAT = None
meta_csv = cfg["paths"].get("metadata_csv")
if meta_csv and os.path.exists(meta_csv):
    try:
        dfm = pd.read_csv(meta_csv)
        if {"id","category"}.issubset(dfm.columns):
            ID2CAT = {str(r.id): str(r.category) for r in dfm.itertuples(index=False)}
    except Exception:
        ID2CAT = None

# --- Optional: load trained head for compatibility rerank ---
use_head = bool(cfg.get("serve", {}).get("use_head", False))
head = None
if use_head and cfg["model"].get("head_weights"):
    try:
        ckpt = torch.load(cfg["model"]["head_weights"], map_location="cpu")
        in_dim = int(ckpt.get("in_dim", X.shape[1]))
        out_dim = int(ckpt.get("out_dim", 256))
        head = ProjectionHead(in_dim=in_dim, out_dim=out_dim)
        head.load_state_dict(ckpt.get("head", ckpt))
        head.eval()
    except FileNotFoundError:
        head = None
        use_head = False

# -------------------- Utils --------------------
def project(v: np.ndarray) -> np.ndarray:
    """Project base embeddings through the compatibility head if available."""
    if head is None:
        return v
    with torch.no_grad():
        t = torch.from_numpy(v.astype("float32"))
        z = head(t)
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-12)
        return z.cpu().numpy().astype("float32")

def mmr_select(cand_vecs: np.ndarray, query_vec: np.ndarray, top_k: int, lam: float = 0.3):
    """Maximal Marginal Relevance for diversity; assumes L2-normalized vectors."""
    N = cand_vecs.shape[0]
    if N == 0:
        return []
    rel = (cand_vecs @ query_vec.T).reshape(-1)  # relevance to query
    selected = []
    remaining = list(range(N))
    # pick best by relevance first
    best = int(np.argmax(rel))
    selected.append(best)
    remaining.remove(best)
    while len(selected) < min(top_k, N) and remaining:
        sims_to_sel = cand_vecs[remaining] @ cand_vecs[selected].T  # [R, |S|]
        div = sims_to_sel.max(axis=1)
        score = lam * rel[remaining] - (1 - lam) * div
        nxt = remaining[int(np.argmax(score))]
        selected.append(nxt)
        remaining.remove(nxt)
    return selected

app = FastAPI(title="Fashion Recsys Search")

class SearchResponse(BaseModel):
    query_id: str
    neighbors: List[str]
    distances: List[float]

# -------------------- Endpoints --------------------
@app.get("/similar", response_model=SearchResponse)
def similar(item_id: str, k: int = 20):
    try:
        idx = ids.index(item_id)
    except ValueError:
        return {"query_id": item_id, "neighbors": [], "distances": []}

    q = X[idx:idx+1].copy()
    if normalize:
        faiss.normalize_L2(q)
    D, I = index.search(q, k)
    neigh = [ids[i] for i in I[0].tolist()]
    dist = D[0].tolist()
    return {"query_id": item_id, "neighbors": neigh, "distances": dist}

@app.get("/complete-look", response_model=SearchResponse)
def complete_look(
    context_ids: str = Query(..., description="comma-separated ids"),
    gap_category: str = "",
    k: int = 20,
    pool: int | None = None,
    use_mmr: bool = True,
    mmr_lambda: float = 0.3
):
    ctx = [c.strip() for c in context_ids.split(",") if c.strip()]
    idxs = [ids.index(c) for c in ctx if c in ids]
    if not idxs:
        return {"query_id": ",".join(ctx), "neighbors": [], "distances": []}

    # Build context vector
    Q = X[idxs].mean(axis=0, keepdims=True)
    if normalize:
        faiss.normalize_L2(Q)

    # Retrieve a candidate pool
    pool_n = pool or int(cfg.get("serve", {}).get("candidates", max(k*5, 100)))
    D, I = index.search(Q, pool_n)
    cand_idx = I[0].astype(int)

    # Category filter (if metadata available)
    if gap_category and ID2CAT is not None:
        filt = [i for i in cand_idx if ID2CAT.get(ids[i], None) == gap_category]
        if len(filt) > 0:
            cand_idx = np.array(filt, dtype=int)

    # Compatibility rerank + optional MMR
    if use_head and head is not None and len(cand_idx) > 0:
        Qp = project(Q)                 # [1, d2]
        Cp = project(X[cand_idx])       # [M, d2]
        if use_mmr and len(cand_idx) > k:
            sel = mmr_select(Cp, Qp, top_k=k, lam=mmr_lambda)
            chosen = cand_idx[sel]
            sims = (Qp @ Cp[sel].T)[0]
            order = np.argsort(-sims)
            cand_idx = chosen[order]
            D = (-sims)[order].reshape(1, -1)
        else:
            sims = (Qp @ Cp.T)[0]
            order = np.argsort(-sims)
            cand_idx = cand_idx[order]
            D = (-sims)[order].reshape(1, -1)
    else:
        # No head: apply MMR in base space (normalized if cosine)
        Cb = X[cand_idx].copy()
        if normalize and len(Cb):
            faiss.normalize_L2(Cb)
        if use_mmr and len(cand_idx) > k:
            sel = mmr_select(Cb, Q, top_k=k, lam=mmr_lambda)
            cand_idx = cand_idx[sel]
            sims = (Cb[sel] @ Q.T).reshape(-1)
            D = (1 - sims)[None, :]

    neigh = [ids[i] for i in cand_idx[:k].tolist()]
    dist = D[0].tolist()[:k] if 'D' in locals() else []
    return {"query_id": ",".join(ctx), "neighbors": neigh, "distances": dist}

