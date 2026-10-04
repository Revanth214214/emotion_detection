import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from src.inference.pipeline.inference_pipeline import InferencePipeline

app = FastAPI(
    title="Emotion Detection API",
    description="API for classifying emotions using PyTorch model",
    version="1.0.0"
)

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the pipeline once at app startup
pipeline = InferencePipeline()

# Mount static files directory for CSS/JS/images assets
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
    return {"status": "healthy", "device": str(pipeline.device)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Accepts an image upload and returns emotion prediction."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    # 1. Save uploaded bytes to a temporary local file
    temp_file_path = f"temp_{file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Pass image path directly into your inference pipeline
        result = pipeline.run(temp_file_path)

        # 3. Clean up the temporary file
        os.remove(temp_file_path)

        return {"status": "success", "prediction": result}

    except Exception as e:
        # Clean up file if inference fails
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)