# FashionRecs

Self-contained reference implementation of a fashion-recommendation pipeline that:

1. Extracts image embeddings with OpenCLIP / SigLIP.
2. Builds a FAISS index for fast nearest-neighbor search.
3. Trains a lightweight projection head to score outfit compatibility (triplet or InfoNCE).
4. Serves recommendations via FastAPI (`index/search_server.py`).

A Colab notebook was used during prototyping; you can adapt it if you need a managed GPU. 
[![Open In C](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ORSHQ4gb5fvByRA1uFapNGYiPvdtpvAE?usp=sharing) 
The repo includes scripts to run everything locally—see below for the minimal workflow and [`train/README.md`](train/README.md) for step-by-step instructions.

## Data Layout

`config.yaml` expects the following structure under `data/`:

```
data/
  raw/
    images/             # image files referenced by metadata_csv
    metadata.csv        # columns: id, path, [category, ...]
  processed/
    train_pairs.csv     # columns: a_id, b_id, label (1 compatible, 0 incompatible)
    embeddings.npy      # generated via train/export_embeddings.py
    ids.json            # aligns row indices to item ids
    head_epoch_last.pt  # saved projection head checkpoint (optional)
    splits.json / outfits.csv (optional extras)
  indexes/
    fashion_embeddings.faiss
```

Adjust `config.yaml` if your paths differ.

## Quickstart

From the repo root:

```bash
# 1. Embed catalog images
python train/export_embeddings.py

# 2. Build FAISS index
python index/build_faiss.py

# 3. Train compatibility head (uses cached embeddings by default)
python train/train_head.py

# 4. Evaluate trained head (optional sanity check)
python train/eval_head.py --cfg config.yaml --samples 2000

# 5. Launch the API server
uvicorn index.search_server:app --host 0.0.0.0 --port 8081
```

See [`train/README.md`](train/README.md) for more detail on each step, configuration knobs, and how to run the optional Colab pipeline. Check `requirements.txt` for dependencies and ensure you have the proper GPU/CPU libraries (e.g., `libomp` on macOS for PyTorch + FAISS).

## Development Notes

- Head checkpoints saved by `train/train_head.py` include `{"head": state_dict, "in_dim": ..., "out_dim": ...}` so `index/search_server.py` can optionally re-rank FAISS candidates.
- Training defaults to cached embeddings; set `train.use_cached_embeddings: false` in `config.yaml` to train directly from images.
- Evaluating positive/negative pair separation is easy via the new `train/eval_head.py` script.
