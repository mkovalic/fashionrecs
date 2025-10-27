"""
Generate data/processed/outfits.csv from item ids that look like "<outfit>_<piece>".
Works with either data/processed/ids.json OR data/raw/metadata.csv (uses whichever is present).

Output CSV schema: outfit_id,item_id
Example row: 100002074,100002074_2
"""
import os, re, json
import pandas as pd

IDS_JSON = "data/processed/ids.json"
METADATA_CSV = "data/raw/metadata.csv"
OUT_CSV = "data/processed/outfits.csv"

pattern = re.compile(r"^([^_]+)_([^_]+)$")

ids = []
if os.path.exists(IDS_JSON):
    ids = json.load(open(IDS_JSON))
elif os.path.exists(METADATA_CSV):
    df = pd.read_csv(METADATA_CSV)
    if "id" not in df.columns:
        raise SystemExit("metadata.csv must contain an 'id' column")
    ids = [str(x) for x in df["id"].tolist()]
else:
    raise SystemExit("No ids.json or metadata.csv found. Generate embeddings first or provide metadata.")

rows = []
skipped = 0
for id_ in ids:
    m = pattern.match(str(id_))
    if not m:
        skipped += 1
        continue
    outfit_id = m.group(1)
    rows.append({"outfit_id": outfit_id, "item_id": str(id_)})

if not rows:
    raise SystemExit("No ids matched the '<outfit>_<piece>' pattern. Nothing to write.")

out = pd.DataFrame(rows).drop_duplicates().sort_values(["outfit_id", "item_id"]).reset_index(drop=True)
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
out.to_csv(OUT_CSV, index=False)
print(f"Wrote {OUT_CSV} with {len(out)} rows. Skipped {skipped} ids that didn't match the pattern.")
