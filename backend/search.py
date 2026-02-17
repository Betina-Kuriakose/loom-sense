import os
from pinecone import Pinecone
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("loom-sense-index")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def find_item(query):
    print(f"🔍 Searching for: '{query}'...")

    try:
        # Generate 768-dim query vector
        res = client.models.embed_content(
            model="gemini-embedding-001",
            contents=query,
            config=types.EmbedContentConfig(
                task_type='RETRIEVAL_QUERY',
                output_dimensionality=768
            )
        )
        query_vector = res.embeddings[0].values

        # Query Pinecone
        results = index.query(vector=query_vector, top_k=1, include_metadata=True)

        if results['matches'] and results['matches'][0]['score'] > 0.3:
            match = results['matches'][0]
            name = match['metadata']['label']
            x, y = match['metadata']['x'], match['metadata']['y']
            
            print(f"\n✅ Found: {name}")
            print(f"📍 Coordinates: ({x}, {y})")
            
            # Simple guidance logic for Alzheimer's
            side = "right" if int(x) > 50 else "left"
            print(f"💡 Guidance: Look toward the {side} side of your desk.")
        else:
            print("❌ Nothing found. Try scanning the desk again with engine.py!")
            
    except Exception as e:
        print(f"❌ Search Error: {e}")

if __name__ == "__main__":
    q = input("What are you looking for? ")
    find_item(q)