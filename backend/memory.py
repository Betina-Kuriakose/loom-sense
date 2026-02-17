import os
from pinecone import Pinecone
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# Setup Clients
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("loom-sense-index")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def save_spatial_data(object_name, x, y):
    try:
        # Generate 768-dim embedding to match Pinecone index
        res = client.models.embed_content(
            model="gemini-embedding-001",
            contents=object_name,
            config=types.EmbedContentConfig(
                task_type='RETRIEVAL_DOCUMENT',
                output_dimensionality=768
            )
        )
        vector = res.embeddings[0].values

        # Save to Pinecone with Metadata
        index.upsert(
            vectors=[{
                "id": f"{object_name}_{x}_{y}", 
                "values": vector,
                "metadata": {"x": x, "y": y, "label": object_name}
            }]
        )
        print(f"✅ Saved {object_name} to cloud memory.")
    except Exception as e:
        print(f"❌ Memory Error: {e}")