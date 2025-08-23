import os, json, time
from pathlib import Path
from typing import List, Dict
import math

from dotenv import load_dotenv
from tqdm import tqdm

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "pawfect-breeds")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = 1536 if "small" in EMBEDDING_MODEL else 3072

DATA_PATH = Path("data/akc-breeds-enriched-fixed.jsonl")
BATCH_SIZE = 100

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

ALLOWED_META_KEYS = {
    "breed", "description", "temperament", "popularity",
    "min_height", "max_height", "min_weight", "max_weight",
    "min_expectancy", "max_expectancy", "group",
    "grooming_frequency_value", "grooming_frequency_category",
    "shedding_value", "shedding_category",
    "energy_level_value", "energy_level_category",
    "trainability_value", "trainability_category",
    "demeanor_value", "demeanor_category",
}

def normalize_scalar(v):
    # Only allow str, bool, int, float; convert NaN/Inf -> None (we'll drop later)
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return float(v)
    if isinstance(v, (int, bool, str)) or v is None:
        return v
    # Fallback: stringify weird scalars
    return str(v)

def normalize_list(lst):
    # Pinecone requires list of STRINGS
    cleaned = []
    for x in lst:
        if x is None:
            continue
        # stringify everything to be safe
        cleaned.append(str(x))
    return cleaned if cleaned else None  # drop empty list

def build_metadata(raw: dict) -> dict:
    meta = {}
    for k in ALLOWED_META_KEYS:
        if k not in raw:
            continue
        v = raw[k]
        if isinstance(v, list):
            v = normalize_list(v)
        else:
            v = normalize_scalar(v)
        if v is None:
            continue  # DROP nulls
        meta[k] = v
    return meta

def ensure_index(name: str, dimension: int):
    # existing = { i["name"] for i in pc.list_indexes()}
    existing = { (i["name"] if isinstance(i, dict) else i.name) for i in pc.list_indexes() }
    if name not in existing:
        pc.create_index(
            name=name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        while True:
            desc = pc.describe_index(name)
            if desc.status["ready"]:
                break
            time.sleep(1)
            
def read_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
    
def embed_texts(texts: List[str]) -> List[List[float]]:
    backoff = 1.0
    while True:
        try:
            resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
            return [d.embedding for d in resp.data]
        except Exception as e:
            print(f"Embedding error: {e}. Retrying in {backoff:.1f}s ...")
            time.sleep(backoff)
            backoff = min(backoff * 2, 10.0)

def chunk_list(items, size):
    for i in range(0, len(items), size):
        yield items[i: i + size]
        
def validate_meta(meta):
    for k, v in meta.items():
        if not (isinstance(v, (str, int, float, bool)) or (isinstance(v, list) and all(isinstance(x, str) for x in v))):
            raise ValueError(f"Invalid meta type for '{k}': {type(v)} -> {v}")

    for _id, _text, _meta in items[:5]:
        validate_meta(_meta)
        
def main():
    assert OPENAI_API_KEY,  "Missing OPENAI_API_KEY"
    assert PINECONE_API_KEY,  "Missing PINECONE_API_KEY"
    assert DATA_PATH.exists(),  f"Missing data file: {DATA_PATH}"
    
    print(f"→ Ensuring Pinecone index '{INDEX_NAME}' ({EMBEDDING_DIM} dims)...")
    ensure_index(INDEX_NAME, EMBEDDING_DIM)
    index = pc.Index(INDEX_NAME)
    
    print(f"→ Loading data from {DATA_PATH} ...")
    rows = read_jsonl(DATA_PATH)
    print("First row:", rows[0])
    print(f"    loaded {len(rows)} breeds.")
    
    items = []
    skipped = 0

    for r in rows:
        _id = r.get("id") or (r.get("breed") or r.get("name") or "unknown").lower().replace(" ", "_")
        text = r.get("match_text") or r.get("description") or r.get("breed")
        if not text or not str(text).strip():
            skipped += 1
            continue

        meta = build_metadata(r)  # <- cleans None and lists

        items.append((_id, str(text), meta))

    print(f"Built {len(items)} items, skipped {skipped}.")
    print("Sample payload meta:", {k: items[0][2].get(k) for k in list(items[0][2].keys())[:6]})

    
    total = 0
    for batch in tqdm(list(chunk_list(items, BATCH_SIZE))):
        ids = [i[0] for i in batch]
        texts = [i[1] for i in batch]
        metas = [i[2] for i in batch]
        
        vectors = embed_texts(texts)
        payload = []
        
        for _id, vec, meta in zip(ids, vectors, metas):
            payload.append({"id": _id, "values": vec, "metadata": meta})
            
        backoff = 1.0

        while True:
            try:
                index.upsert(vectors=payload)
                break
            except Exception as e:
                print(f"Upsert error: {e}. Retrying in {backoff:.1f}s ...")
                time.sleep(backoff)
                backoff = min(backoff * 2, 10.0)

        total += len(batch)

    print(f"✅ Done. Upserted {total} vectors into '{INDEX_NAME}'.")
    
if __name__ == "__main__":
    main()