import os, sys, json, re
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

# local modules
from parser import parse_query_to_json
from filters import build_pinecone_filter
from reranker import rerank_with_llm

load_dotenv()

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME       = os.getenv("PINECONE_INDEX_NAME", "pawfect-breeds")
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# clients
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

def embed(text: str):
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding

# --- heuristic constraints from casual phrases ---
def derive_constraints_from_text(text: str) -> dict:
    t = text.lower()
    c = {}

    # Size
    if "x-small" in t or "toy" in t or "extra small" in t:
        c["size"] = "x-small"
    elif "small" in t:
        c["size"] = "small"
    elif "medium" in t or "mid-size" in t or "midsize" in t:
        c["size"] = "medium"
    elif "large" in t:
        c["size"] = "large"
    elif "x-large" in t or "extra large" in t or "giant" in t:
        c["size"] = "x-large"

    # Energy / exercise
    if any(p in t for p in ["calm", "low energy", "couch", "doesn’t need too much exercise",
                             "doesn't need too much exercise", "low-maintenance energy"]):
        c["energy_max"] = 0.4
    if any(p in t for p in ["active", "very active", "outdoorsy", "run", "runner", "hike", "swim", "athletic"]):
        c["energy_min"] = 0.6

    # Trainability
    if any(p in t for p in ["easy to train", "highly trainable", "eager to please", "intelligent"]):
        c["trainability_min"] = 0.6

    # Hypoallergenic / shedding
    if "hypoallergenic" in t or "allergy" in t or "low shedding" in t or "low-shedding" in t:
        c["shedding_max"] = 0.4

    # Grooming burden
    if "low-maintenance" in t or "low maintenance" in t or "minimal grooming" in t:
        c["grooming_max"] = 0.5

    # Apartment living (often implies lower energy / smaller size)
    if "apartment" in t or "small apartment" in t:
        c.setdefault("energy_max", 0.4)
        # if no explicit size, lightly bias toward small
        c.setdefault("size", "small")

    return c


def infer_simple_tags(text: str) -> dict:
    """Cheap heuristics to ensure obvious tags are included even if parser misses them."""
    t = text.lower()
    tags = {"colors": [], "coat_texture": [], "ear_shape": []}
    if re.search(r"\bwhite\b", t):
        tags["colors"].append("white")
    if "fluffy" in t or "plush" in t:
        tags["coat_texture"].append("fluffy")
    if "pointed ear" in t or "pointy ear" in t or "erect ear" in t or "upright ear" in t:
        tags["ear_shape"].extend(["pointed", "erect"])
    return tags

COLOR_RELAX_MAP = {
    "white": ["white", "cream", "ivory"],
    "black": ["black"],
    "red":   ["red"],
    # extend later as needed
}

def relax_filter_constraints(pc_filter: dict, stage: int) -> dict:
    """
    stage 0 = strict (as-is)
    stage 1 = drop coat_texture
    stage 2 = drop ear_shape
    stage 3 = expand colors via COLOR_RELAX_MAP (e.g., white -> [white, cream, ivory])
    stage 4 = drop colors
    stage 5 = keep only size/weight (final fallback)
    """
    f = dict(pc_filter or {})
    if stage >= 1:
        f.pop("coat_texture", None)
    if stage >= 2:
        f.pop("ear_shape", None)
    if stage >= 3 and "colors" in f:
        vals = f["colors"].get("$in", [])
        expanded = []
        for v in vals:
            expanded += COLOR_RELAX_MAP.get(v, [v])
        if expanded:
            f["colors"] = {"$in": sorted(set(s.lower() for s in expanded))}
    if stage >= 4:
        f.pop("colors", None)
    if stage >= 5:
        # keep only size/weight keys
        f = {k: v for k, v in f.items() if k in ("max_weight", "min_weight")}
    return f


def _as_list(v):
    if v is None:
        return []
    if isinstance(v, (list, tuple, set)):
        return list(v)
    return [v]

def merge_list_constraints(base: dict, extra: dict, keys=("colors","coat_texture","ear_shape")):
    base = dict(base or {})
    extra = dict(extra or {})
    out = dict(base)

    for k in keys:
        have = _as_list(base.get(k))
        add  = _as_list(extra.get(k))

        # normalize + lowercase
        have = [str(x).strip().lower() for x in have if str(x).strip()]
        add  = [str(x).strip().lower() for x in add if str(x).strip()]

        merged = list({*have, *add})  # de-dupe, order not guaranteed
        if merged:
            out[k] = merged
    return out

def query_breeds(user_text: str, use_rerank: bool = True, recall_k: int = 30, final_k: int = 3):
    parsed = parse_query_to_json(user_text)

    # 1) merge list-like tags inferred earlier (colors/ears/texture)
    parsed["constraints"] = merge_list_constraints(parsed.get("constraints") or {}, infer_simple_tags(user_text))

    # 2) merge numeric/categorical heuristics
    heur = derive_constraints_from_text(user_text)
    parsed["constraints"] = {**heur, **parsed["constraints"], **{k:v for k,v in parsed.get("constraints", {}).items() if v is not None}}

    base_filter = build_pinecone_filter(parsed["constraints"])
    pref = parsed.get("preference_text") or user_text
    vec = embed(pref)

    matches = []
    applied_filter = None
    # try up to 6 relaxation stages
    for stage in range(0, 6):
        trial_filter = relax_filter_constraints(base_filter or {}, stage)
        res = index.query(
            vector=vec,
            top_k=max(parsed.get("top_k", final_k), recall_k),
            include_metadata=True,
            filter=trial_filter if trial_filter else None
        )
        if res.get("matches"):
            matches = res["matches"]
            applied_filter = trial_filter
            break

    print(f"\n🔍 {user_text}")
    print("Filter:", json.dumps(applied_filter or base_filter or {}, ensure_ascii=False))

    if not matches:
        print("No results even after progressive relaxation. Try removing a constraint (e.g., color) or widening size.")
        return

    # Rerank on the found pool
    ordered = rerank_with_llm(user_text, parsed, matches, top_k=final_k) if use_rerank else matches[:final_k]

    for m in ordered:
        md = m["metadata"]
        title = md.get("breed","?")
        fit = m.get("_llm_fit_score")
        fit_str = f"{fit:.3f}" if isinstance(fit, (int,float)) else f"{m.get('score',0):.3f}"
        print(f"\n⭐ {title}  fit={fit_str}")
        if m.get("_llm_hard_misses"):
            print("   misses:", ", ".join(m["_llm_hard_misses"]))
        if m.get("_llm_why"):
            print("   why:", " · ".join(m["_llm_why"]))
        desc = (md.get("description") or "")[:220].rstrip()
        print("   " + desc + ("..." if len(desc) == 220 else ""))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_q = " ".join(sys.argv[1:])
    else:
        user_q = "I want a white fluffy dog with pointed ears, medium size, intelligent and family-friendly"
        user_q = "I'm looking for a small, low-maintenance, hypoallergenic dog. I live in a small apartment, so I prefer a calm, cuddly breed that doesn’t need too much exercise."
        user_q = "I'm an outdoorsy and active person looking for a medium to large dog that loves to run, hike, and swim. I want an adventurous, athletic companion to share outdoor activities with"
    query_breeds(user_q, use_rerank=True, recall_k=30, final_k=3)
