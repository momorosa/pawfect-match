import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "pawfect-breeds")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# clients
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

def embed(text: str):
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding

def query_breeds(user_text: str, top_k: int = 3):
    print(f"\n🔍 Query: {user_text}\n")
    vec = embed(user_text)
    res = index.query(vector=vec, top_k=top_k, include_metadata=True)
    for match in res["matches"]:
        md = match["metadata"]
        breed = md.get("breed", "Unknown")
        desc = md.get("description", "")[:200] + "..."
        score = match["score"]
        print(f"⭐ {breed} (score={score:.3f})")
        print(f"   {desc}\n")
        
if __name__ == "__main__":
    query_breeds("I'm looking for a medium-sized dog with a white coat. It should be alert, easy to train, and highly intelligent.")
    query_breeds("I'm looking for a small, low-maintenance, hypoallergenic dog. I live in a small apartment, so I prefer a calm, cuddly breed that doesn’t need too much exercise.")
    query_breeds("I'm an outdoorsy and active person looking for a medium to large dog that loves to run, hike, and swim. I want an adventurous, athletic companion to share outdoor activities with.")