from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from tensorflow import keras
from keras.models import load_model
from PIL import Image
import numpy as np
import io
import logging
import os


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="Dogs vs Cats Classifier API",
    description="API для классификации изображений собак и кошек",
    version="1.0.0"
)

MODEL_PATH = "models/dogs_cats_cnn.keras"
model = None


@app.on_event("startup")
async def load_model_start():
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            logger.warning(f"Model file not found at {MODEL_PATH}")
            return
            
        logger.info(f"Loading model from {MODEL_PATH}...")
        model = load_model(MODEL_PATH, compile=False)
        model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

        logger.info(f"Model successfully loaded from {MODEL_PATH}")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model = None


CLASS_NAMES = ["Cat", "Dog"]


def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise HTTPException(status_code=400, detail="Invalid image file")


@app.get("/active")
def activity_check():
    return {
        "status": "active" if model is not None else "model_not_loaded",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    }


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Please check if model file exists.")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        contents = await file.read()
        
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")
        
        processed_image = preprocess_image(contents)
        prediction = model.predict(processed_image, verbose=0)
        
        predicted_class = CLASS_NAMES[int(prediction[0][0] > 0.5)]
        confidence_score = float(prediction[0][0])
        confidence = confidence_score if predicted_class == "Dog" else float(1 - confidence_score)
        
        logger.info(f"Prediction for {file.filename}: {predicted_class} ({confidence:.2f})")
        
        return JSONResponse(content={
            "filename": file.filename,
            "class": predicted_class,
            "confidence": round(confidence, 4),
            "raw_prediction": float(confidence_score)
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction")
