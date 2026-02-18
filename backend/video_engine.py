import os
import time
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv
from memory import save_spatial_data

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def process_spatial_video(video_path):
    print(f"📁 Uploading {video_path} to Gemini...")
    
    # 1. Upload the file
    video_file = client.files.upload(file=video_path)
    print(f"⏳ File uploaded. Processing: {video_file.name}...")

    # 2. Wait for the video to be ready (Active state)
    while video_file.state.name == "PROCESSING":
        print(".", end="", flush=True)
        time.sleep(2)
        video_file = client.files.get(name=video_file.name)

    if video_file.state.name == "FAILED":
        raise Exception("Video processing failed.")

    print("\n🎬 Video is ready. Analyzing spatial layout...")

    # 3. Request spatial grounding and object detection
    # We ask for a "panoramic" understanding of the video
    prompt = """
    Watch this video scan of a room. 
    Identify key household objects (e.g., phone, mug, medicine, glasses).
    For each object, provide the coordinates where it is located at the END of the video.
    Return only a JSON list: [{"name": "item", "x": 500, "y": 700}]
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash", # Use 2.0 or 1.5 for stable video support
        contents=[
            types.Part.from_uri(file_uri=video_file.uri, mime_type=video_file.mime_type),
            prompt
        ],
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )

    # 4. Save results to Pinecone
    try:
        detected_objects = json.loads(response.text)
        print(f"🔍 AI found {len(detected_objects)} items in the video.")
        
        for obj in detected_objects:
            save_spatial_data(obj.get('name'), obj.get('x'), obj.get('y'))
            
        print("\n✨ Video scan complete. Spatial memory updated.")
        
        # Clean up: Delete the file from Google cloud after processing
        client.files.delete(name=video_file.name)
        
    except Exception as e:
        print(f"❌ Analysis Error: {e}")

if __name__ == "__main__":
    # Ensure you have a short video file (e.g., 'room_scan.mp4') in your folder
    process_spatial_video("room_scan.mp4")