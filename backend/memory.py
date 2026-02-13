import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "loom-sense-index"

# Create the index if it doesn't exist
if index_name not in pc.list_indexes().names():
    print(f"Creating index: {index_name}...")
    pc.create_index(
        name=index_name,
        dimension=768, # Gemini's embedding size is usually 768
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print("✅ Index created!")
else:
    print(f"✅ Index '{index_name}' already exists and is ready.")