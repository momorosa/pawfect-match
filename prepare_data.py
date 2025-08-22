import pandas as pd
import json
import re

# --- Helpers ---
def slugify(text):
    import re
    s = re.sub(r"\s+", "_", str(text).strip().lower())
    s = re.sub(r"[^a-z0-9_]", "", s)
    return s

# --- Load CSV ---
df = pd.read_csv("data/akc-data-latest.csv")

# --- Rename ---
if "Unnamed: 0" in df.columns:
    df = df.rename(columns={ "Unnamed: 0": "breed" })
    
# --- Build records with id + match_text ---
records = []
for _, row in df.iterrows():
    kv_pairs = []
    for col, val in row.items():
        if pd.isna(val):
            continue
        kv_pairs.append(f"{col}: {val}")
    match_text = " | ".join(kv_pairs)
    
    rec = row.to_dict()
    rec["id"] = slugify(rec.get("breed", "unknown"))
    rec["match_text"] = match_text
    records.append(rec)
    
# --- Save as JSONL ---
out_path = "data/akc-breeds-enriched-fixed.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
        
print(f"File saved as {out_path}")

# --- Quick Verification ---
print("\nHere are the first 3 lines:")
with open(out_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        print(line.strip())