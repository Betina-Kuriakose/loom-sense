import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv
from memory import save_spatial_data

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def analyze_workspace(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Error: {image_path} not found!")
        return

    print(f"🚀 Scanning {image_path}...")
    
    with open(image_path, "rb") as f:
        image_data = f.read()

    # Get JSON results from Gemini
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=image_data, mime_type="image/png"),
            "Identify objects on the desk. Return a JSON list: [{'name': 'item', 'x': 50, 'y': 20}]"
        ],
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )

    try:
        detected_objects = json.loads(response.text)
        for obj in detected_objects:
            save_spatial_data(obj.get('name'), obj.get('x'), obj.get('y'))
        print("\n✨ Desk scanning complete.")
    except Exception as e:
        print(f"❌ Analysis Error: {e}")

if __name__ == "__main__":
    analyze_workspace("desk.png")