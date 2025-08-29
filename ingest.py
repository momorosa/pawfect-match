import os, json, time, re, math
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv
from tqdm import tqdm

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# ── env ────────────────────────────────────────────────────────────────────────
load_dotenv()

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME       = os.getenv("PINECONE_INDEX_NAME", "pawfect-breeds")
PINECONE_CLOUD   = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION  = os.getenv("PINECONE_REGION", "us-east-1")

EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM    = 1536 if "small" in EMBEDDING_MODEL else 3072

DATA_PATH = Path("data/akc-breeds-enriched-fixed.jsonl")
BATCH_SIZE = 100

# ── clients ────────────────────────────────────────────────────────────────────
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# ── enrichment vocab (lowercase) ───────────────────────────────────────────────
COLOR_WORDS = {
    "white","black","red","blue","fawn","cream","grey","gray","silver","gold",
    "liver","chocolate","brown","tan","brindle","merle","sable","tricolor",
    "bi-color","bicolor","parti","wheaten","black and tan","red and white",
    "salt-and-pepper","pepper and salt"
}
EAR_SYNONYMS = {
    "pointed": ["pointed ears","prick ears","erect ears","upright ears","erect"],
    "floppy":  ["floppy ears","drop ears","pendulous ears"],
    "semi-erect": ["semi-erect","button ears","rose ears","v-shaped ears"],
}
COAT_TYPE_PHRASES = [
    "double coat","single coat","wire coat","rough coat","smooth coat",
    "long coat","short coat","medium coat","silky coat","harsh coat",
    "curly coat","wavy coat","corded coat","cords","dense undercoat",
    "soft undercoat","outer coat","undercoat",
]
COAT_TEXTURE_CLUES = {
    "fluffy": ["fluffy","plush","full coat","thick coat","long coat","dense undercoat","double coat"]
}
TAIL_SHAPES = [
    "curled tail","sickle tail","rolled tail","plumed tail","feathered tail",
    "bobtail","docked tail","ring tail","otter tail","saber tail","carrot-shaped tail",
]
FACE_FEATURES = ["beard","mustache","moustache","whiskers","bushy eyebrows","wrinkles","wrinkled"]
BUILD_WORDS = ["compact","muscular","powerful","stocky","robust","slender","athletic","agile","elegant"]

# ── helpers ────────────────────────────────────────────────────────────────────
def _norm_list(lst):
    out, seen = [], set()
    for x in lst:
        s = str(x).strip().lower()
        if s and s not in seen:
            out.append(s); seen.add(s)
    return out

def extract_features(text: str) -> dict:
    """Return dict with optional keys: colors, ear_shape, coat_texture, coat_type, tail_shape, face_features, build."""
    t = (text or "").lower()

    # colors
    found_colors = set()
    for c in COLOR_WORDS:
        if " " in c:
            if c in t: found_colors.add(c)
        else:
            if re.search(rf"\b{re.escape(c)}\b", t): found_colors.add(c)

    # ear shapes
    ear_labels = {lab for lab, phrases in EAR_SYNONYMS.items() if any(p in t for p in phrases)}

    # coat texture (coarse)
    coat_textures = {lab for lab, clues in COAT_TEXTURE_CLUES.items() if any(c in t for c in clues)}

    # coat type phrases / tails / face / build
    coat_types = {p for p in COAT_TYPE_PHRASES if p in t}
    tails      = {p for p in TAIL_SHAPES if p in t}
    faces      = {p for p in FACE_FEATURES if p in t}
    builds     = {p for p in BUILD_WORDS if re.search(rf"\b{re.escape(p)}\b", t)}

    out = {}
    if found_colors:  out["colors"]        = _norm_list(found_colors)
    if ear_labels:    out["ear_shape"]     = _norm_list(ear_labels)
    if coat_textures: out["coat_texture"]  = _norm_list(coat_textures)
    if coat_types:    out["coat_type"]     = _norm_list(coat_types)
    if tails:         out["tail_shape"]    = _norm_list(tails)
    if faces:         out["face_features"] = _norm_list(faces)
    if builds:        out["build"]         = _norm_list(builds)
    return out

ALLOWED_META_KEYS = {
    "breed","description","temperament","popularity",
    "min_height","max_height","min_weight","max_weight",
    "min_expectancy","max_expectancy","group",
    "grooming_frequency_value","grooming_frequency_category",
    "shedding_value","shedding_category",
    "energy_level_value","energy_level_category",
    "trainability_value","trainability_category",
    "demeanor_value","demeanor_category",
}

def normalize_scalar(v):
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v): return None
        return float(v)
    if isinstance(v, (int, bool, str)) or v is None:
        return v
    return str(v)

def normalize_list(lst):
    cleaned = [str(x) for x in lst if x is not None]
    cleaned = [s for s in cleaned if s.strip()]
    return cleaned if cleaned else None

def build_metadata(raw: dict) -> dict:
    meta = {}
    for k in ALLOWED_META_KEYS:
        if k not in raw: continue
        v = raw[k]
        if isinstance(v, list): v = normalize_list(v)
        else: v = normalize_scalar(v)
        if v is None: continue
        meta[k] = v
    return meta

def validate_meta(meta: dict):
    if not isinstance(meta, dict):
        raise ValueError(f"Metadata must be dict, got {type(meta)}")
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            continue
        if isinstance(v, list) and all(isinstance(x, str) for x in v):
            continue
        raise ValueError(f"Invalid meta type for '{k}': {type(v)} -> {v}")

def ensure_index(name: str, dimension: int):
    existing = {(i["name"] if isinstance(i, dict) else i.name) for i in pc.list_indexes()}
    if name not in existing:
        pc.create_index(
            name=name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
    # wait until ready
    while True:
        desc = pc.describe_index(name)
        if desc.status.get("ready"): break
        print("   …waiting for index to be ready")
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
            time.sleep(backoff); backoff = min(backoff * 2, 10.0)

def chunk_list(items, size):
    for i in range(0, len(items), size):
        yield items[i:i+size]

# ── main ───────────────────────────────────────────────────────────────────────
def main():
    assert OPENAI_API_KEY,  "Missing OPENAI_API_KEY"
    assert PINECONE_API_KEY, "Missing PINECONE_API_KEY"
    assert DATA_PATH.exists(), f"Missing data file: {DATA_PATH}"

    print(f"→ Ensuring Pinecone index '{INDEX_NAME}' ({EMBEDDING_DIM} dims)...")
    ensure_index(INDEX_NAME, EMBEDDING_DIM)
    index = pc.Index(INDEX_NAME)

    print(f"→ Loading data from {DATA_PATH} ...")
    rows = read_jsonl(DATA_PATH)
    print("First row:", rows[0])
    print(f"   loaded {len(rows)} breeds.")

    items, skipped = [], 0
    for r in rows:
        _id = r.get("id") or (r.get("breed") or r.get("name") or "unknown").lower().replace(" ", "_")
        text = r.get("match_text") or r.get("description") or r.get("breed")
        if not text or not str(text).strip():
            skipped += 1; continue

        meta = build_metadata(r)
        # enrich with extracted tags from description + match_text
        source_text = f"{r.get('description','')} {r.get('match_text','')}"
        extra = extract_features(source_text)
        for k, v in extra.items():
            if v: meta[k] = v  # lists of strings

        validate_meta(meta)  # fail fast if bad types
        items.append((_id, str(text), meta))

    print(f"Built {len(items)} items, skipped {skipped}.")
    if not items:
        raise RuntimeError("No items built — check JSONL contents/path.")
    print("Sample payload meta:", {k: items[0][2].get(k) for k in list(items[0][2].keys())[:6]})

    print("→ Embedding and upserting to Pinecone ...")
    total = 0
    for batch in tqdm(list(chunk_list(items, BATCH_SIZE))):
        ids   = [i[0] for i in batch]
        texts = [i[1] for i in batch]
        metas = [i[2] for i in batch]

        vectors = embed_texts(texts)
        payload = [{"id": _id, "values": vec, "metadata": meta}
                   for _id, vec, meta in zip(ids, vectors, metas)]

        backoff = 1.0
        while True:
            try:
                index.upsert(vectors=payload)
                break
            except Exception as e:
                print(f"Upsert error: {e}. Retrying in {backoff:.1f}s ...")
                time.sleep(backoff); backoff = min(backoff * 2, 10.0)

        total += len(batch)

    print(f"✅ Done. Upserted {total} vectors into '{INDEX_NAME}'.")

if __name__ == "__main__":
    main()
