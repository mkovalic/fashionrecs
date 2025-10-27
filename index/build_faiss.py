import json
import yaml
import numpy as np
import faiss


def build_index(vectors: np.ndarray, metric: str = "cosine"):
    d = vectors.shape[1]
    if metric == "cosine":
        # cosine similarity = inner product on L2-normalized vectors
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(d)
    elif metric == "l2":
        index = faiss.IndexFlatL2(d)
    else:
        raise ValueError("metric must be 'cosine' or 'l2'")
    index.add(vectors)
    return index


def main(cfg_path="config.yaml"):
    cfg = yaml.safe_load(open(cfg_path))
    X = np.load(cfg["paths"]["embeddings_npy"]).astype("float32")
    index = build_index(X, cfg["index"]["metric"])
    faiss.write_index(index, cfg["paths"]["faiss_index"])

if __name__ == "__main__":
    main()