import os, json
from openai import OpenAI


PARSER_MODEL = "gpt-4o-mini"

PARSER_SYS = """You extract structured dog-breed matching preferences from casual text.
Return STRICT JSON with keys: constraints, preference_text, top_k. Follow the schema. No extra text."""

def get_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY in environment")
    return OpenAI(api_key=api_key)

def parse_query_to_json(user_text: str) -> dict:
    client = get_client()
    resp = client.chat.completions.create(
        model=PARSER_MODEL,
        response_format={"type":"json_object"},
        messages=[
            {"role":"system","content":PARSER_SYS},
            {"role":"user","content":user_text}
        ],
        temperature=0
    )
    raw = resp.choices[0].message.content
    try:
        data = json.loads(raw)
    except Exception:
        data = {"constraints":{}, "preference_text":user_text, "top_k":3}
    data.setdefault("constraints", {})
    data.setdefault("preference_text", user_text)
    data["top_k"] = int(data.get("top_k") or 3)
    return data


# def _ensure_list(v):
#     if v is None: return []
#     return v if isinstance(v, list) else [v]

# parsed = parse_query_to_json(user_text)
# con = parsed.get("constraints", {})
# con["ear_shape"] = _ensure_list(con.get("ear_shape"))
# con["colors"] = _ensure_list(con.get("colors"))
# con["coat_texture"] = _ensure_list(con.get("coat_texture"))

