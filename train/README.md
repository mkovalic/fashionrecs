# Training & Embedding Workflow

The repository now includes everything required to train the embedding encoder outputs, build the FAISS index, and fit the compatibility head locally. The steps below describe the supported, scriptable flow; if you still prefer Colab for convenience, you can adapt those notebook cells to mirror each section here (see the short note at the end), but it’s no longer required.

## Prerequisites

- Python 3.10+ with packages listed in `requirements.txt` (key ones: `torch`, `torchvision`, `open-clip-torch`, `faiss-cpu`, `pandas`, `numpy`, `tqdm`, `pyyaml`, `fastapi`).
- Images + metadata:
  - `paths.images`: directory containing images referenced in the metadata CSV.
  - `paths.metadata_csv`: CSV with at least `id` and `path` columns (`path` is relative to `paths.images`).
- Pair file `paths.pairs_csv` containing `a_id`, `b_id`, `label` (1 = compatible, 0 = incompatible).

## 1. Generate Base Embeddings

Local script (`train/export_embeddings.py`) handles the entire pass:

```bash
python train/export_embeddings.py  # uses config.yaml paths
```

Outputs:

- `paths.embeddings_npy` – `float32` array `[N, D]` of normalized encoder features.
- `paths.ids_json` – list of string ids aligned with the rows of the embedding matrix.

Colab alternative: use the exported notebook cells you pasted (they read `data/raw/metadata.csv`, run OpenCLIP, and write the same `.npy`/`.json` artifacts). Upload/download those two files plus the metadata CSV when moving between Colab and local.

## 2. Build the FAISS Index

Run the helper script:

```bash
python index/build_faiss.py        # honors config.yaml
```

It loads `paths.embeddings_npy`, normalizes when `index.metric: "cosine"`, builds the appropriate Flat index, and writes `paths.faiss_index`. The logic matches the Colab cell you exported, so there’s no need to duplicate it.

The API server (`index/search_server.py`) reads the `.faiss` file, `ids.json`, and optionally the trained head for reranking.

## 3. Train the Compatibility Head

`train/train_head.py` now supports two modes (see `train/dataset.py` and `train/losses.py` for the dataset/loss definitions it imports):

- **Cached embeddings (default)** – fastest. It loads `paths.embeddings_npy`/`paths.ids_json` and the `train.pairs_csv` labels to sample triplets or pairs without touching images.
- **Image-based** – set `train.use_cached_embeddings: false` if you need to recompute encoder features on the fly (requires GPU VRAM).

Run:

```bash
python train/train_head.py            # uses config.yaml
# or specify a different config
python train/train_head.py path/to/config.yaml
```

Key config knobs:

| Field | Purpose |
| --- | --- |
| `model.backbone`, `model.pretrained` | Passed to OpenCLIP when `use_cached_embeddings: false`. |
| `model.embed_dim` | `"auto"` infers from the backbone; otherwise supply an integer. |
| `model.head_out_dim` | Output dimension of `ProjectionHead`. Stored in the checkpoint for the search server. |
| `model.head_weights` | Destination path (defaults to `data/processed/head_epoch{N}.pt`). |
| `train.loss` | `"triplet"` (default) or `"infonce"`. |
| `train.use_cached_embeddings` | Toggle between cached vs. image loaders. |
| `train.neg_pool_ratio` | Optional downsampling for negative pairs when building triplets. |
| `train.margin`, `train.lr_head`, `train.weight_decay`, `train.batch_size`, `train.epochs` | Standard optimizer/training parameters. |

Each epoch saves a checkpoint shaped like:

```python
{"head": state_dict, "in_dim": embed_dim, "out_dim": head_out_dim}
```

`index/search_server.py` expects exactly this structure when reranking candidates.

## 4. Quick Evaluation

After training, you can reproduce the notebook’s sanity check by sampling positive/negative pairs from `train_pairs.csv`, projecting with the saved head, and comparing cosine similarity (the helpers in `train/dataset.py` make it easy to spin up a quick script if you’d like to automate this). This step is optional but helps confirm the head separates compatible looks.

## Optional: Colab Pipeline

All of the above steps originated from a Colab prototype. If you need a quick GPU (e.g., no local CUDA), you can run the notebook version by uploading the same inputs (`config.yaml`, pairs/metadata CSVs, embeddings/ids if already computed) and exporting the resulting `embeddings.npy`, `ids.json`, `head_epoch_last.pt`, and FAISS index back into `data/processed/`. Just make sure any tweaks you make there are mirrored in the scripts so the repo stays reproducible on its own.

---

When pushing updates, ensure `config.yaml` paths reflect the committed artifacts (embeddings, ids, FAISS index, head weights) so `index/search_server.py` and `train/train_head.py` remain reproducible.
