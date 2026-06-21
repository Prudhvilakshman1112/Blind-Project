import io
import base64
import numpy as np
import cv2
import uvicorn
from PIL import Image
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config_demo import API_HOST, API_PORT, API_CORS_ORIGINS
from vision_module import ObjectDetector
from caption_module_demo import GeminiCaptioner, SpatialReasoningNLP

app = FastAPI(title="Blind-Project Demo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=API_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
detector = None
captioner = None
reasoner = None

@app.on_event("startup")
def load_models():
    global detector, captioner, reasoner
    print("Loading YOLO and Gemini Demo modules...")
    detector = ObjectDetector()
    captioner = GeminiCaptioner()
    reasoner = SpatialReasoningNLP()
    print("Models loaded successfully.")

class FrameRequest(BaseModel):
    image: str  # Base64 encoded JPEG

def decode_base64_image(base64_string: str) -> np.ndarray:
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    return np.array(image)[:, :, ::-1]  # Convert RGB to BGR for OpenCV

def background_caption_task(frame: np.ndarray):
    """Runs Gemini captioning in the background. It will return early if interval hasn't passed."""
    captioner.caption(frame)

@app.post("/analyze")
async def analyze_frame(request: FrameRequest, background_tasks: BackgroundTasks):
    try:
        frame = decode_base64_image(request.image)
        
        # Fast YOLO detection (sync)
        detections = detector.detect(frame)
        
        # Trigger background Gemini captioning
        background_tasks.add_task(background_caption_task, frame)
        
        # Use the most recent caption available
        raw_caption = captioner.last_caption
        
        description = reasoner.build_description(raw_caption, detections)
        
        alerts = []
        # Get priority alerts
        for det in detections:
            if det.get("is_high_priority") or det.get("in_danger_zone"):
                alert_text = reasoner.build_danger_alert(det)
                alerts.append(alert_text)
                
        return {
            "caption": description,
            "alerts": alerts,
            "detections": detections
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("api_demo:app", host=API_HOST, port=API_PORT, reload=False)
