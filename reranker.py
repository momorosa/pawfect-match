import os, json
from openai import OpenAI

RERANK_MODEL = "gpt-4o-mini"

RERANK_SYS = """You rank dog breeds using ONLY the provided candidate metadata and the parsed user constraints.
Rules:
- DO NOT invent facts. If a field is missing, treat it as unknown and do not fabricate.
- Enforce hard constraints first (size/weight range, shedding_max/min, energy range, trainability_min, grooming_max, colors, ear_shape, coat_texture).
- Penalize missing hard tags: if a tag is explicitly requested (e.g., colors=['white']) and the candidate has tags that do not include it, mark a hard_miss.
- After hard constraints, use preference_text to break ties (e.g., 'family-friendly', 'watchdog ok', 'intelligent', 'likes training').
- Output STRICT JSON with this shape:
{
  "ranking":[
    {"id":"<breed or id>", "fit_score":0..1, "hard_misses":["..."], "soft_notes":["..."], "why":["short bullet","short bullet"]},
    ...
  ]
}
Scoring guidance:
- Start from 1.0 and subtract 0.25 for each hard_miss (color/ear/texture/size/shedding/energy/trainability/grooming).
- If multiple hard_misses, the item may drop below 0.5 quickly; keep scores in [0,1].
- If there are no hard_misses and preference_text matches temperament/description well, boost up to [0.8..1.0].
- Never list candidates not provided.
"""

def _compact_candidates(matches):
    out = []
    for m in matches:
        md = m.get("metadata", {})
        out.append({
            "id": md.get("breed") or m.get("id"),
            "pinecone_score": m.get("score"),
            "min_weight": md.get("min_weight"),
            "max_weight": md.get("max_weight"),
            "shedding_value": md.get("shedding_value"),
            "energy_level_value": md.get("energy_level_value"),
            "trainability_value": md.get("trainability_value"),
            "grooming_frequency_value": md.get("grooming_frequency_value"),
            "colors": md.get("colors"),
            "ear_shape": md.get("ear_shape"),
            "coat_texture": md.get("coat_texture"),
            "temperament": md.get("temperament"),
            "description": (md.get("description") or "")[:300]
        })
    return out

def rerank_with_llm(user_query: str, parsed: dict, matches: list, top_k: int = 3):
    """
    matches: Pinecone matches (list of dicts with 'metadata','score')
    returns: ordered list of dicts from the original matches (length <= top_k)
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    payload = {
        "user_query": user_query,
        "constraints": parsed.get("constraints", {}),
        "preference_text": parsed.get("preference_text", user_query),
        "candidates": _compact_candidates(matches)
    }

    resp = client.chat.completions.create(
        model=RERANK_MODEL,
        response_format={"type":"json_object"},
        temperature=0,
        messages=[
            {"role":"system","content": RERANK_SYS},
            {"role":"user","content": json.dumps(payload, ensure_ascii=False)}
        ]
    )
    data = json.loads(resp.choices[0].message.content or "{}")
    ranking = data.get("ranking", [])

    # map back to original matches
    id_to_match = {}
    for m in matches:
        md = m.get("metadata", {})
        key = md.get("breed") or m.get("id")
        if key:
            id_to_match[key] = m

    ordered = []
    for r in sorted(ranking, key=lambda x: x.get("fit_score", 0), reverse=True):
        m = id_to_match.get(r.get("id"))
        if m:
            # attach LLM reasons for your UI
            m["_llm_fit_score"] = r.get("fit_score")
            m["_llm_why"] = r.get("why", [])
            m["_llm_hard_misses"] = r.get("hard_misses", [])
            ordered.append(m)
        if len(ordered) >= top_k:
            break

    # Fallback if LLM returns nothing usable
    if not ordered:
        ordered = matches[:top_k]
    return ordered
