from fastapi import FastAPI, File, UploadFile, HTTPException
from src.inference.pipeline.inference_pipeline import InferencePipeline
import os
import shutil

app = FastAPI(
    title="Emotion Detection API",
    description="API for classifying emotions using PyTorch model",
    version="1.0.0"
)

# Initialize the pipeline once at app startup
pipeline = InferencePipeline()

@app.get("/")
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
        result = pipeline.run_pipeline(temp_file_path) # Or pipeline.predict(temp_file_path) depending on your method name

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