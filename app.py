import logging
import os
import time

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.inference.pipeline.inference_pipeline import InferencePipeline

logger = logging.getLogger("emotion_api")

# --- Settings (override with environment variables) -------------------------
MAX_UPLOAD_BYTES = int(float(os.getenv("MAX_UPLOAD_MB", "5")) * 1024 * 1024)
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.5"))
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()
]
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
API_VERSION = "1.1.0"

app = FastAPI(
    title="Emotion Detection API",
    description="API for classifying emotions using PyTorch model",
    version=API_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Initialize the pipeline once at app startup
pipeline = InferencePipeline()

if os.path.exists("src/static"):
    app.mount("/static", StaticFiles(directory="src/static"), name="static")


@app.get("/")
def serve_frontend():
    """Serves the live webcam frontend UI."""
    index_path = os.path.join("src", "static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Frontend index.html not found in src/static/", "status": "healthy"}


@app.get("/health")
def health_check():
    """Health check endpoint to verify container status."""
    return {
        "status": "healthy",
        "device": str(pipeline.device),
        "version": API_VERSION,
        "min_confidence": MIN_CONFIDENCE,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Accepts an image upload and returns the emotion for each detected face.

    status is "success" when at least one face was found, "no_face" otherwise.
    Predictions below MIN_CONFIDENCE are labelled "Uncertain".
    """
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG or WebP image.")

    # Read at most limit+1 bytes so oversized uploads are rejected cheaply.
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).",
        )

    # Decode in memory: no temp files, safe for concurrent requests.
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    try:
        start = time.perf_counter()
        results = pipeline.run_frame(frame, min_confidence=MIN_CONFIDENCE)
        inference_ms = round((time.perf_counter() - start) * 1000, 1)
    except Exception:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Inference failed.")

    return {
        "status": "success" if results else "no_face",
        "num_faces": len(results),
        "inference_ms": inference_ms,
        "prediction": results,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
