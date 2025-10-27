"""
Create positive/negative pairs from outfit groupings.
Input: data/processed/outfits.csv with columns [outfit_id,item_id]
Output: data/processed/train_pairs.csv with columns [a_id,b_id,label]
"""
import pandas as pd
import numpy as np
from itertools import combinations

in_path = "data/processed/outfits.csv"
out_path = "data/processed/train_pairs.csv"

outfits = pd.read_csv(in_path)
by_outfit = outfits.groupby("outfit_id")["item_id"].apply(list)

pairs = []
for items in by_outfit:
    # positives: all unordered pairs within an outfit
    for a,b in combinations(items, 2):
        pairs.append((a,b,1))

all_ids = outfits["item_id"].unique()
all_ids_set = set(all_ids)

# generate simple negatives by random mismatching
rng = np.random.default_rng(42)
for items in by_outfit:
    for a in items:
        # pick a random negative id not in the same outfit
        neg = rng.choice(list(all_ids_set - set(items)))
        pairs.append((a, neg, 0))

pd.DataFrame(pairs, columns=["a_id","b_id","label"]).to_csv(out_path, index=False)
print(f"wrote {out_path}")