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
    # 1) Parse to structured JSON
    parsed = parse_query_to_json(user_text)

    # 2) Merge simple inferred tags (helps with color/texture/ears)
    parsed["constraints"] = merge_list_constraints(parsed.get("constraints") or {}, infer_simple_tags(user_text))

    # 3) Build Pinecone filter
    pc_filter = build_pinecone_filter(parsed["constraints"])

    # 4) Embed preference_text (fallback to original query)
    pref = parsed.get("preference_text") or user_text
    vec = embed(pref)

    # 5) Query Pinecone with generous recall
    res = index.query(
        vector=vec,
        top_k=max(parsed.get("top_k", final_k), recall_k),
        include_metadata=True,
        filter=pc_filter
    )
    matches = res.get("matches", [])

    # 6) Optional LLM rerank
    if use_rerank and matches:
        ordered = rerank_with_llm(user_text, parsed, matches, top_k=final_k)
    else:
        ordered = matches[:final_k]

    # Pretty print
    print(f"\n🔍 {user_text}")
    print("Filter:", json.dumps(pc_filter or {}, ensure_ascii=False))
    if not ordered:
        print("No results. Consider relaxing constraints (e.g., drop color or widen size range).")
        return

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
    query_breeds(user_q, use_rerank=True, recall_k=30, final_k=3)
