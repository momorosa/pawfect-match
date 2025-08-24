import os, re
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "pawfect-breeds")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# --- helpers ----------------------------------------------------------

# rough size buckets using MAX WEIGHT (your metadata is in kilograms)
# xs  <= 4.5 kg   (~10 lb)
# sm  4.5–11.5 kg (~10–25 lb)
# md  11.5–22.7 kg (~25–50 lb)
# lg  22.7–40.8 kg (~50–90 lb)
# xl  > 40.8 kg
SIZE_BUCKETS = {
    "x-small": {"max_weight": {"$lte": 4.5}},
    "extra small": {"max_weight": {"$lte": 4.5}},
    "toy": {"max_weight": {"$lte": 4.5}},
    "small": {"max_weight": {"$lte": 11.5}},
    "medium": {"max_weight": {"$gte": 11.5, "$lte": 22.7}},
    "large": {"max_weight": {"$gte": 22.7, "$lte": 40.8}},
    "x-large": {"max_weight": {"$gte": 40.8}},
    "extra large": {"max_weight": {"$gte": 40.8}},
    "giant": {"max_weight": {"$gte": 40.8}},
}

def embed(text: str):
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding

def parse_constraints(user_text: str):
    t = user_text.lower()
    flt = {}

    # size (prefer the last mentioned size word)
    size_hits = [k for k in SIZE_BUCKETS.keys() if k in t]
    if size_hits:
        picked = size_hits[-1]
        flt.update(SIZE_BUCKETS[picked])

    # shedding (dataset has shedding_value 0..1)
    # low/mild/light -> <= 0.4 ; high/heavy -> >= 0.6
    if re.search(r"\b(low[-\s]?shedding|hypoallergenic|low shed|less shedding)\b", t):
        flt["shedding_value"] = {"$lte": 0.4}
    elif re.search(r"\b(high[-\s]?shedding|heavy shed|lots of shedding)\b", t):
        flt["shedding_value"] = {"$gte": 0.6}

    # energy (energy_level_value 0..1)
    if re.search(r"\b(low energy|calm|chill|couch|low-maintenance)\b", t):
        flt["energy_level_value"] = {"$lte": 0.4}
    elif re.search(r"\b(high energy|very active|runner|athletic|working)\b", t):
        flt["energy_level_value"] = {"$gte": 0.6}
    elif re.search(r"\b(moderate|medium energy|somewhat active)\b", t):
        flt["energy_level_value"] = {"$gte": 0.4, "$lte": 0.6}

    # trainability (trainability_value 0..1)
    if re.search(r"\b(easy to train|highly trainable|eager to please|obedient)\b", t):
        flt["trainability_value"] = {"$gte": 0.6}
    elif re.search(r"\b(stubborn|independent|hard to train)\b", t):
        flt["trainability_value"] = {"$lte": 0.4}

    # barking tolerance (optional; dataset has no direct bark scale, skip)
    # grooming: you have both *_category (string) and *_value (0..1). Here’s a simple value filter:
    if re.search(r"\b(low grooming|low maintenance grooming|minimal grooming)\b", t):
        flt["grooming_frequency_value"] = {"$lte": 0.4}
    elif re.search(r"\b(high grooming|frequent grooming|daily grooming)\b", t):
        flt["grooming_frequency_value"] = {"$gte": 0.6}

    # remove any empty dicts (Pinecone filter must not have empty objects)
    for k in list(flt.keys()):
        if isinstance(flt[k], dict) and not flt[k]:
            del flt[k]

    return flt

def query_breeds(user_text: str, top_k: int = 3):
    constraints = parse_constraints(user_text)
    vec = embed(user_text)
    print("\n🔍 Query:", user_text)
    if constraints:
        print("🔎 Applying filter:", constraints)

    res = index.query(
        vector=vec,
        top_k=max(top_k, 3),
        include_metadata=True,
        filter=constraints if constraints else None
    )

    for m in res["matches"]:
        md = m["metadata"]
        breed = md.get("breed", "Unknown")
        size_hint = ""
        if "max_weight" in md and "min_weight" in md:
            size_hint = f"  (≈ {md['min_weight']:.1f}-{md['max_weight']:.1f} kg)"
        print(f"\n⭐ {breed} — score={m['score']:.3f}{size_hint}")
        desc = (md.get("description") or "")[:220].rstrip()
        print("   " + desc + ("..." if len(desc) == 220 else ""))

if __name__ == "__main__":
    # Try a few:
    query_breeds("I'm looking for a medium-sized dog with a white coat. It should be alert, easy to train, and highly intelligent.")
    query_breeds("I'm looking for a small, low-maintenance, hypoallergenic dog. I live in a small apartment, so I prefer a calm, cuddly breed that doesn’t need too much exercise.")
    query_breeds("I'm an outdoorsy and active person looking for a medium to large dog that loves to run, hike, and swim. I want an adventurous, athletic companion to share outdoor activities with.")