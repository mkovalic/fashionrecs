import os
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.encoders import FrozenOpenCLIP
from models.heads import ProjectionHead
from train.dataset import (
    ImageTable,
    PairDataset,
    PairFromEmbeddings,
    TripletDataset,
    TripletFromEmbeddings,
)
from train.losses import InfoNCELoss, triplet_loss


def main(cfg_path: str = "config.yaml"):
    cfg = yaml.safe_load(open(cfg_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_cached = should_use_cached_embeddings(cfg)
    loss_mode = cfg["train"]["loss"]

    if use_cached:
        ds, embed_dim = build_cached_dataset(cfg)
        encoder = None
    else:
        encoder = FrozenOpenCLIP(cfg["model"]["backbone"], cfg["model"].get("pretrained"))
        ds, embed_dim = build_image_dataset(cfg, encoder)

    dl = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        drop_last=(loss_mode == "triplet"),
        num_workers=cfg["train"].get("num_workers", 4),
        pin_memory=True,
    )

    head = ProjectionHead(in_dim=embed_dim, out_dim=cfg["model"].get("head_out_dim", 256)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg["train"]["lr_head"], weight_decay=cfg["train"]["weight_decay"])
    criterion = InfoNCELoss() if loss_mode == "infonce" else None
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    if encoder is not None:
        encoder.model.to(device)
        encoder.model.eval()

    head.train()
    for epoch in range(cfg["train"]["epochs"]):
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}")
        running = 0.0
        steps = 0
        for batch in pbar:
            opt.zero_grad(set_to_none=True)
            loss = forward_step(batch, device, encoder, head, loss_mode, criterion, cfg)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += float(loss)
            steps += 1
            pbar.set_postfix(loss=running / max(steps, 1))

        save_head(
            head,
            embed_dim,
            cfg["model"].get("head_out_dim", 256),
            cfg["model"].get("head_weights", f"data/processed/head_epoch{epoch+1}.pt"),
        )


def should_use_cached_embeddings(cfg: dict) -> bool:
    train_cfg = cfg.get("train", {})
    if not train_cfg.get("use_cached_embeddings", True):
        return False
    paths = cfg.get("paths", {})
    emb = paths.get("embeddings_npy")
    ids = paths.get("ids_json")
    return bool(emb and ids and os.path.exists(emb) and os.path.exists(ids))


def build_cached_dataset(cfg: dict):
    paths = cfg["paths"]
    loss_mode = cfg["train"]["loss"]
    if loss_mode == "triplet":
        ds = TripletFromEmbeddings(
            paths["pairs_csv"],
            paths["ids_json"],
            paths["embeddings_npy"],
            neg_pool_ratio=cfg["train"].get("neg_pool_ratio", 1.0),
        )
    else:
        ds = PairFromEmbeddings(paths["pairs_csv"], paths["ids_json"], paths["embeddings_npy"])
    return ds, ds.embed_dim


def build_image_dataset(cfg: dict, encoder: FrozenOpenCLIP):
    transform = encoder.get_preprocess()
    table = ImageTable(cfg["paths"]["images"], cfg["paths"]["metadata_csv"])
    loss_mode = cfg["train"]["loss"]
    if loss_mode == "triplet":
        ds = TripletDataset(
            cfg["paths"]["pairs_csv"],
            table,
            transform,
            neg_pool_ratio=cfg["train"].get("neg_pool_ratio", 1.0),
        )
    else:
        ds = PairDataset(cfg["paths"]["pairs_csv"], table, transform)

    embed_dim = cfg["model"].get("embed_dim", "auto")
    if embed_dim == "auto":
        embed_dim = encoder.embed_dim
    return ds, int(embed_dim)


def forward_step(batch, device, encoder, head, loss_mode, criterion, cfg):
    amp_enabled = torch.cuda.is_available()
    if loss_mode == "triplet":
        a, p, n = batch
        a = a.to(device)
        p = p.to(device)
        n = n.to(device)
        if encoder is not None:
            with torch.no_grad():
                a = encoder(a)
                p = encoder(p)
                n = encoder(n)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            az, pz, nz = head(a), head(p), head(n)
            loss = triplet_loss(az, pz, nz, margin=cfg["train"].get("margin", 0.3))
    else:
        a, b, _ = batch
        a = a.to(device)
        b = b.to(device)
        if encoder is not None:
            with torch.no_grad():
                a = encoder(a)
                b = encoder(b)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            az, bz = head(a), head(b)
            loss = criterion(az, bz)
    return loss


def save_head(head, in_dim: int, out_dim: int, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"head": head.state_dict(), "in_dim": in_dim, "out_dim": out_dim}, path)


if __name__ == "__main__":
    main()
