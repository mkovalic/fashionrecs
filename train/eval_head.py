"""
Quick compatibility-head evaluation over cached embeddings.

Usage:
    python train/eval_head.py --cfg config.yaml --samples 2000
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

# Ensure repository root is on sys.path so `models` imports work when invoked from any cwd
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.heads import ProjectionHead
from train.dataset import PairFromEmbeddings


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate compatibility head on cached embeddings.")
    parser.add_argument("--cfg", default="config.yaml", help="Path to config file.")
    parser.add_argument("--samples", type=int, default=2000, help="Number of positive/negative pairs to evaluate.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for evaluation loader.")
    parser.add_argument("--device", default="cpu", help="Device for the projection head (cpu or cuda).")
    return parser.parse_args()


def load_head(cfg: dict, embed_dim: int, device: torch.device) -> ProjectionHead:
    ckpt_path = cfg["model"].get("head_weights")
    if not ckpt_path or not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Head weights not found at {ckpt_path}. Train the head first.")
    ckpt = torch.load(ckpt_path, map_location=device)
    in_dim = int(ckpt.get("in_dim", embed_dim))
    out_dim = int(ckpt.get("out_dim", cfg["model"].get("head_out_dim", 256)))
    head = ProjectionHead(in_dim=in_dim, out_dim=out_dim).to(device)
    state = ckpt.get("head", ckpt)
    head.load_state_dict(state)
    head.eval()
    return head


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.cfg))

    dataset = PairFromEmbeddings(
        cfg["paths"]["pairs_csv"],
        cfg["paths"]["ids_json"],
        cfg["paths"]["embeddings_npy"],
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    head = load_head(cfg, dataset.embed_dim, device)

    target = args.samples
    pos_scores: list[float] = []
    neg_scores: list[float] = []

    for a, b, labels in loader:
        if len(pos_scores) >= target and len(neg_scores) >= target:
            break

        a = a.to(device)
        b = b.to(device)
        with torch.no_grad():
            az = head(a)
            bz = head(b)
            sims = F.cosine_similarity(az, bz)

        labels = labels.to(device)
        for sim, lbl in zip(sims.tolist(), labels.tolist()):
            if lbl >= 0.5:
                if len(pos_scores) < target:
                    pos_scores.append(sim)
            else:
                if len(neg_scores) < target:
                    neg_scores.append(sim)
        if math.isclose(len(pos_scores), target) and math.isclose(len(neg_scores), target):
            break

    if not pos_scores or not neg_scores:
        raise RuntimeError("Insufficient samples to compute metrics. Check pairs CSV or increase dataset size.")

    pos_mean = sum(pos_scores) / len(pos_scores)
    neg_mean = sum(neg_scores) / len(neg_scores)

    print(f"Samples used  |  positives: {len(pos_scores)}  negatives: {len(neg_scores)}")
    print(f"Mean cosine   |  positives: {pos_mean:.4f}  negatives: {neg_mean:.4f}")
    print(f"Margin        |  pos - neg : {pos_mean - neg_mean:.4f}")


if __name__ == "__main__":
    main()
