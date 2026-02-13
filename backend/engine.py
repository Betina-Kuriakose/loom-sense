import os
import json
from google import genai
from google.genai import types  # type: ignore
from dotenv import load_dotenv # type: ignore
from memory import save_spatial_data

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

def analyze_workspace(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Error: {image_path} not found!")
        return

    print(f"🚀 Scanning {image_path} with Gemini...")
    
    with open(image_path, "rb") as f:
        image_data = f.read()

    # ONE clean call to Gemini with the correct JSON configuration
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=image_data, mime_type="image/png"),
            "Identify objects on the desk. Return a JSON list of objects. "
            "Format: [{'name': 'item', 'x': 50, 'y': 20}]"
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json" 
        )
    )

    print("\n--- Processing Results ---")
    
    try:
        # Convert the JSON string from Gemini into a Python list
        detected_objects = json.loads(response.text)
        
        for obj in detected_objects:
            # Send to Pinecone
            save_spatial_data(
                object_name=obj.get('name', 'unknown'),
                x=obj.get('x', 0),
                y=obj.get('y', 0)
            )
        print("\n✅ All items saved to memory!")
            
    except json.JSONDecodeError:
        print("❌ Error: Gemini didn't return valid JSON.")
    except Exception as e:
        print(f"❌ An error occurred: {e}")

analyze_workspace("desk.png")