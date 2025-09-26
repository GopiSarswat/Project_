from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import cv2
import os
import logging
import urllib.request
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------
# Configuration
# ------------------------------
MODEL_PATH = "cattle_model.tflite"
CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
CASCADE_PATH = "haarcascade_frontalface_default.xml"

# Confidence thresholds
PREDICTION_THRESHOLD = 1.0  # Lower threshold to show more results
MAX_PREDICTIONS = 2

# ------------------------------
# Download Haar Cascade
# ------------------------------
def download_haar_cascade():
    if os.path.exists(CASCADE_PATH):
        return True
    logger.info("📥 Downloading Haar cascade...")
    try:
        urllib.request.urlretrieve(CASCADE_URL, CASCADE_PATH)
        return os.path.exists(CASCADE_PATH)
    except:
        return False

cascade_downloaded = download_haar_cascade()

# ------------------------------
# Load Models
# ------------------------------
def load_models():
    models = {}
    
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        
        models['interpreter'] = tf.lite.Interpreter(model_path=MODEL_PATH)
        models['interpreter'].allocate_tensors()
        models['input_details'] = models['interpreter'].get_input_details()
        models['output_details'] = models['interpreter'].get_output_details()
        logger.info("✅ Cattle breed model loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to load cattle breed model: {e}")
        raise
    
    models['face_cascade'] = None
    if cascade_downloaded and os.path.exists(CASCADE_PATH):
        try:
            cascade = cv2.CascadeClassifier(CASCADE_PATH)
            if not cascade.empty():
                models['face_cascade'] = cascade
                logger.info("✅ Human detection model loaded successfully!")
        except:
            logger.warning("⚠️ Human detection disabled")
    
    return models

models = load_models()

# ------------------------------
# Define class names (39 breeds)
# ------------------------------
class_names = [
    "Alambadi", "Amritmahal", "Ayrshire", "Banni", "Bargur", "Bhadawari",
    "Brown Swiss", "Deoni", "Gir", "Guernsey", "Hallikan", "Hariana",
    "Holstein Friesian", "Jaffrabadi", "Jersey", "Kangayam", "Kankrej",
    "Kasangod", "Khillari", "Krishna Valley", "Malnad gidda", "Mehsana",
    "Murrah", "Nagori", "Nagpuri", "Nimari", "Ongole", "Pulikulam",
    "Rathi", "Red Dane", "Red Sindhi", "Sahival", "Surti", "Tharparkan",
    "Toda", "Umblachery", "Vechur"
]

# ------------------------------
# FastAPI app setup
# ------------------------------
app = FastAPI(
    title="🐄 Cattle Breed Classifier API",
    description="Cattle Breed Identification API",
    version="1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# Helper functions
# ------------------------------
def detect_humans(image_array: np.ndarray) -> tuple:
    if models.get('face_cascade') is None:
        return False, "Human detection unavailable"
    try:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        faces = models['face_cascade'].detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return len(faces) > 0, f"Found {len(faces)} human face(s)"
    except:
        return False, "Human detection error"

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

def get_model_predictions(image_array: np.ndarray) -> np.ndarray:
    interpreter = models['interpreter']
    input_details = models['input_details']
    output_details = models['output_details']
    
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

def filter_predictions(predictions: np.ndarray) -> List[Dict]:
    sorted_indices = np.argsort(predictions)[::-1]
    results = []
    
    for idx in sorted_indices:
        confidence = float(predictions[idx]) * 100
        if confidence >= PREDICTION_THRESHOLD and len(results) < MAX_PREDICTIONS:
            results.append({
                "breed": class_names[idx],
                "confidence": round(confidence, 2)
            })
    
    return results

# ------------------------------
# API endpoints
# ------------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🐄 Cattle Breed Classifier API</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🐄 Cattle Breed Classifier API</h1>
            <p>Upload an image to identify cattle breed</p>
            <p><a href="/docs">API Documentation</a></p>
        </div>
    </body>
    </html>
    """

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Identify cattle breed from uploaded image
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, "File must be an image (JPEG or PNG)")
    
    # Check file size
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > 10 * 1024 * 1024:
        raise HTTPException(400, "Image size too large. Maximum 10MB allowed.")
    
    try:
        # Read image
        image_bytes = await file.read()
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        if pil_image.format not in ['JPEG', 'PNG']:
            raise HTTPException(400, "Only JPEG and PNG images are supported")
        
        image_array = np.array(pil_image.convert("RGB"))
        
        # Check for humans
        if models.get('face_cascade') is not None:
            has_humans, human_msg = detect_humans(image_array)
            if has_humans:
                raise HTTPException(400, f"Human detected: {human_msg}. Upload cattle images only.")
        
        # Preprocess and get model predictions
        processed_img = preprocess_image(image_bytes)
        raw_predictions = get_model_predictions(processed_img)
        
        # Filter predictions (top 2 only)
        predictions = filter_predictions(raw_predictions)
        
        if not predictions:
            raise HTTPException(400, "No valid predictions generated")
        
        # Return in exact requested format
        response = {
            "success": True,
            "predictions": predictions,
            "top_prediction": predictions[0] if predictions else None
        }
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(500, f"Prediction failed: {str(e)}")

@app.get("/breeds")
async def get_breeds():
    return {
        "total_breeds": len(class_names),
        "breeds": class_names
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")