import os
from google import genai
from google.genai import types 
from dotenv import load_dotenv # type: ignore

load_dotenv()

# You NEED these two lines below to initialize the AI
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

def analyze_workspace(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Error: {image_path} not found in this folder!")
        return

    print(f"🚀 Scanning {image_path} with Gemini...")
    
    with open(image_path, "rb") as f:
        image_data = f.read()

    response = client.models.generate_content(
model="gemini-2.5-flash",  # <--- Use this updated 2026 model name
        contents=[
            types.Part.from_bytes(
                data=image_data,
                mime_type="image/png" # Set to PNG based on your previous error log
            ),
            "List every object on this desk. For each, give a coordinate (x, y) "
            "where (0,0) is top-left and (100,100) is bottom-right."
        ]
    )
    print("\n--- Spatial Analysis ---")
    print(response.text)

# Ensure the filename here matches your actual file!
analyze_workspace("desk.png")