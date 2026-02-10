import os
from google import genai
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv() # Load your API keys

# 1. Setup Gemini
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 2. Setup Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("loom-sense-v1")

def scan_and_remember(image_path):
    # 'Look' at the image
    with open(image_path, "rb") as f:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=["Identify objects in this workspace and their relative locations.", f.read()]
        )
    
    description = response.text
    print(f"Gemini sees: {description}")

    # 'Embed' and save to Pinecone (Simplified RAG)
    # In the full project, you'd use client.models.embed_content here
    # For now, we'll just store the text metadata
    index.upsert(vectors=[("desk_1", [0.1]*768, {"text": description})]) 
    print("Memory stored in Pinecone!")

# Run it! (Replace with a real path to a photo of your desk)
# scan_and_remember("my_desk.jpg")
