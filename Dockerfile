# 1. Base Image: Use an official lightweight Python runtime
FROM python:3.12-slim

# 2. Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# 3. Set the working directory inside the container
WORKDIR /app

# 4. Install system dependencies required by OpenCV and facial detection
# 4. Install system dependencies required by OpenCV and facial detection
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy requirements file first to leverage Docker layer caching
COPY requirements.txt .

# 6. Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --extra-index-url https://pypi.org/simple -r requirements.txt

# 7. Copy the entire project into the container
COPY . .

# 8. Expose port 8000 for FastAPI
EXPOSE 8000

# 9. Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]