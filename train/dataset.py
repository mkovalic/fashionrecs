"""
Datasets used during embedding export and head training.

Includes both image-based datasets (for end-to-end fine-tuning using the
encoder) and fast datasets that operate on cached embeddings, mirroring
the Colab notebook workflow.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageTable:
    """Lightweight index over metadata CSV for image lookups by id."""

    def __init__(self, images_root: str | Path, metadata_csv: str | Path):
        self.images_root = Path(images_root)
        self.df = pd.read_csv(metadata_csv)
        required = {"id", "path"}
        if not required.issubset(self.df.columns):
            raise ValueError(f"metadata file must contain columns {required}")
        self.id2path: Dict[str, Path] = {
            str(row.id): self.images_root / str(row.path) for row in self.df.itertuples(index=False)
        }

    def __len__(self) -> int:
        return len(self.df)

    def load(self, item_id: str) -> Image.Image:
        path = self.id2path.get(str(item_id))
        if path is None:
            raise KeyError(f"unknown item id {item_id}")
        return Image.open(path).convert("RGB")


class PairDataset(Dataset):
    """
    Returns pairs of images (anchor, other) plus label for contrastive or InfoNCE loss.
    CSV must contain columns: a_id, b_id, label.
    """

    def __init__(self, pairs_csv: str | Path, table: ImageTable, transform):
        self.df = pd.read_csv(pairs_csv)
        required = {"a_id", "b_id", "label"}
        if not required.issubset(self.df.columns):
            raise ValueError(f"pairs file must contain columns {required}")
        self.table = table
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        a_img = self.transform(self.table.load(row.a_id))
        b_img = self.transform(self.table.load(row.b_id))
        label = torch.tensor(float(row.label), dtype=torch.float32)
        return a_img, b_img, label


class TripletDataset(Dataset):
    """
    Triplet sampler that reads positives/negatives from the pair CSV and returns images.
    """

    def __init__(self, pairs_csv: str | Path, table: ImageTable, transform, neg_pool_ratio: float = 1.0):
        self.df = pd.read_csv(pairs_csv)
        required = {"a_id", "b_id", "label"}
        if not required.issubset(self.df.columns):
            raise ValueError(f"pairs file must contain columns {required}")
        self.table = table
        self.transform = transform

        pos = self.df[self.df.label == 1][["a_id", "b_id"]].values.tolist()
        neg = self.df[self.df.label == 0][["a_id", "b_id"]].values.tolist()
        self.pos_pairs: List[Tuple[str, str]] = [(str(a), str(b)) for a, b in pos if str(a) in table.id2path and str(b) in table.id2path]
        self.neg_pairs: List[Tuple[str, str]] = [(str(a), str(b)) for a, b in neg if str(a) in table.id2path and str(b) in table.id2path]
        if neg_pool_ratio < 1.0 and self.neg_pairs:
            k = max(1, int(len(self.neg_pairs) * neg_pool_ratio))
            self.neg_pairs = random.sample(self.neg_pairs, k)

        self.all_ids = list(table.id2path.keys())

    def __len__(self) -> int:
        return len(self.pos_pairs)

    def __getitem__(self, idx: int):
        a_id, p_id = self.pos_pairs[idx]
        a_img = self.transform(self.table.load(a_id))
        p_img = self.transform(self.table.load(p_id))
        n_id = self._sample_negative(a_id, p_id)
        n_img = self.transform(self.table.load(n_id))
        return a_img, p_img, n_img

    def _sample_negative(self, a_id: str, p_id: str) -> str:
        if self.neg_pairs:
            _, neg_id = random.choice(self.neg_pairs)
            return neg_id
        return random.choice(self.all_ids)


class _EmbeddingBase(Dataset):
    """Common utilities for embedding-based datasets."""

    def __init__(self, ids_json: str | Path, embeddings_npy: str | Path):
        self.ids: List[str] = [str(x) for x in json.load(open(ids_json))]
        self.id2row = {id_: i for i, id_ in enumerate(self.ids)}
        self.embeddings = np.load(embeddings_npy).astype("float32")
        self.embed_dim = self.embeddings.shape[1]
        if self.embeddings.shape[0] != len(self.ids):
            raise ValueError("embeddings and ids length mismatch")

    def _vec(self, idx: int) -> torch.Tensor:
        v = torch.from_numpy(self.embeddings[idx])
        return v / (v.norm() + 1e-12)


class TripletFromEmbeddings(_EmbeddingBase):
    """
    Triplet dataset that operates directly on cached embeddings (Colab workflow).
    """

    def __init__(
        self,
        pairs_csv: str | Path,
        ids_json: str | Path,
        embeddings_npy: str | Path,
        neg_pool_ratio: float = 1.0,
    ):
        super().__init__(ids_json, embeddings_npy)
        self.df = pd.read_csv(pairs_csv)
        pos = self.df[self.df.label == 1][["a_id", "b_id"]].values.tolist()
        neg = self.df[self.df.label == 0][["a_id", "b_id"]].values.tolist()
        self.pos_pairs = [(str(a), str(b)) for a, b in pos if a in self.id2row and b in self.id2row]
        self.neg_pairs = [(str(a), str(b)) for a, b in neg if a in self.id2row and b in self.id2row]
        if neg_pool_ratio < 1.0 and self.neg_pairs:
            k = max(1, int(len(self.neg_pairs) * neg_pool_ratio))
            self.neg_pairs = random.sample(self.neg_pairs, k)
        self.all_rows = np.array(list(self.id2row.values()), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.pos_pairs)

    def __getitem__(self, idx: int):
        a_id, p_id = self.pos_pairs[idx]
        a_idx = self.id2row[a_id]
        p_idx = self.id2row[p_id]
        n_idx = self._sample_negative(a_id, p_id)
        return self._vec(a_idx), self._vec(p_idx), self._vec(n_idx)

    def _sample_negative(self, a_id: str, p_id: str) -> int:
        if self.neg_pairs:
            na_id, nb_id = random.choice(self.neg_pairs)
            if random.random() < 0.5 and na_id in (a_id, p_id):
                cand = nb_id
            else:
                cand = nb_id
            if cand in self.id2row:
                return self.id2row[cand]
        return int(np.random.choice(self.all_rows))


class PairFromEmbeddings(_EmbeddingBase):
    """Pairs of cached embeddings + label for InfoNCE or contrastive losses."""

    def __init__(self, pairs_csv: str | Path, ids_json: str | Path, embeddings_npy: str | Path):
        super().__init__(ids_json, embeddings_npy)
        self.df = pd.read_csv(pairs_csv)
        mask = self.df[["a_id", "b_id"]].map(lambda x: str(x) in self.id2row)
        self.df = self.df[mask.all(axis=1)].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        a = self._vec(self.id2row[str(row.a_id)])
        b = self._vec(self.id2row[str(row.b_id)])
        label = torch.tensor(float(row.label), dtype=torch.float32)
        return a, b, label


__all__ = [
    "ImageTable",
    "PairDataset",
    "TripletDataset",
    "TripletFromEmbeddings",
    "PairFromEmbeddings",
]
