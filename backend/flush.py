import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("loom-sense-index")

print("🧹 Clearing old memory...")
index.delete(delete_all=True)
print("✨ Memory is now empty and ready for fresh data!")