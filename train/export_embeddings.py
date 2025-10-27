import json
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import pandas as pd

from models.encoders import FrozenOpenCLIP

class ImageOnlyDataset(Dataset):
    def __init__(self, images_root, metadata_csv, transform):
        self.df = pd.read_csv(metadata_csv)
        self.images_root = images_root
        self.t = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(f"{self.images_root}/{row.path}").convert("RGB")
        return self.t(img), str(row.id)

@torch.no_grad()
def main(cfg_path="config.yaml"):
    cfg = yaml.safe_load(open(cfg_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc = FrozenOpenCLIP(cfg["model"]["backbone"], cfg["model"]["pretrained"]).to(device)
    ds = ImageOnlyDataset(cfg["paths"]["images"], cfg["paths"]["metadata_csv"], enc.get_preprocess())
    dl = DataLoader(ds, batch_size=256, num_workers=4, pin_memory=True)

    ids = []
    embs = []
    for imgs, batch_ids in tqdm(dl):
        imgs = imgs.to(device)
        feats = enc(imgs).cpu().numpy()
        embs.append(feats)
        ids.extend(batch_ids)

    embs = np.concatenate(embs, axis=0).astype("float32")

    np.save(cfg["paths"]["embeddings_npy"], embs)
    json.dump(ids, open(cfg["paths"]["ids_json"], "w"))

if __name__ == "__main__":
    main()