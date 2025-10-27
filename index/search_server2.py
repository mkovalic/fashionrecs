import json
import yaml
from typing import List

import faiss
import numpy as np
from fastapi import FastAPI, Query
from pydantic import BaseModel

# Optional rerank with trained head
import torch
from models.heads import ProjectionHead

# Load config, ids, FAISS
cfg = yaml.safe_load(open("config.yaml"))
ids = json.load(open(cfg["paths"]["ids_json"]))
index = faiss.read_index(cfg["paths"]["faiss_index"])
metric = cfg["index"]["metric"]

if metric == "cosine":
    normalize = True
else:
    normalize = False

# Load embeddings matrix for query-by-id (you can store separately if huge)
X = np.load(cfg["paths"]["embeddings_npy"]).astype("float32")

# --- Optional: load trained head for compatibility rerank ---
use_head = bool(cfg.get("serve", {}).get("use_head", False))
head = None
if use_head and cfg["model"].get("head_weights"):
    try:
        ckpt = torch.load(cfg["model"]["head_weights"], map_location="cpu")
        in_dim = int(ckpt.get("in_dim", X.shape[1]))
        out_dim = int(ckpt.get("out_dim", 256))
        head = ProjectionHead(in_dim=in_dim, out_dim=out_dim)
        head.load_state_dict(ckpt["head"]) if "head" in ckpt else head.load_state_dict(ckpt)
        head.eval()
    except FileNotFoundError:
        head = None
        use_head = False


def project(v: np.ndarray) -> np.ndarray:
    """Project base embeddings through the compatibility head if available.
    v: [N,D] float32 numpy (assumed L2-normalized already for cosine).
    returns: [N,D2] float32 numpy L2-normalized
    """
    if head is None:
        return v
    with torch.no_grad():
        t = torch.from_numpy(v.astype("float32"))
        z = head(t)
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-12)
        return z.cpu().numpy().astype("float32")

app = FastAPI(title="Fashion Recsys Search")

class SearchResponse(BaseModel):
    query_id: str
    neighbors: List[str]
    distances: List[float]

@app.get("/similar", response_model=SearchResponse)
def similar(item_id: str, k: int = 20):
    # Query by existing item id
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

# "Complete the look" with optional compatibility rerank
@app.get("/complete-look", response_model=SearchResponse)
def complete_look(context_ids: str = Query(..., description="comma-separated ids"), gap_category: str = "bottom", k: int = 20):
    ctx = [c.strip() for c in context_ids.split(",") if c.strip()]
    idxs = [ids.index(c) for c in ctx if c in ids]
    if not idxs:
        return {"query_id": ",".join(ctx), "neighbors": [], "distances": []}

    # Average context embeddings (replace with attention pooling later)
    Q = X[idxs].mean(axis=0, keepdims=True)
    if normalize:
        faiss.normalize_L2(Q)

    # Pull a larger candidate set for rerank
    pool = int(cfg.get("serve", {}).get("candidates", max(k*5, 100)))
    D, I = index.search(Q, pool)
    cand_idx = I[0].astype(int)

    # TODO: filter by gap_category using metadata if available

    # Optional compatibility rerank in the learned head space
    if use_head and head is not None:
        Qp = project(Q)                 # [1, d2]
        Cp = project(X[cand_idx])       # [pool, d2]
        # cosine similarity in projected space
        sims = (Qp @ Cp.T)[0]
        order = np.argsort(-sims)
        cand_idx = cand_idx[order]
        D = (-sims)[order].reshape(1, -1)  # pseudo distance for output

    neigh = [ids[i] for i in cand_idx[:k].tolist()]
    dist = D[0].tolist()[:k]
    return {"query_id": ",".join(ctx), "neighbors": neigh, "distances": dist}