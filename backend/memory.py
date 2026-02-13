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
    def save_spatial_data(object_name, x, y):
        index = pc.Index(index_name)
        
        # In a real RAG app, we would use an embedding model here.
        # For now, we store the coordinates as metadata.
        # We use a dummy vector [0.1, 0.2...] because Pinecone requires one.
        
        dummy_vector = [0.1] * 768
        index.upsert(
            vectors=[
                {
                    "id": object_name, 
                    "values": dummy_vector, 
                    "metadata": {"x": x, "y": y, "label": object_name}
                }
            ]
        )
        print(f"Saved {object_name} to memory at ({x}, {y})")