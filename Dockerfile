# Start with the python-slim base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
# Set the Hugging Face cache directory to a predictable location
ENV HF_HOME=/app/huggingface_cache

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip

# --- THE DEFINITIVE FIX: Pre-download the model ---
# Create a dummy script to download the model files to the cache
RUN python -c "from transformers import AutoImageProcessor, AutoModelForImageClassification; \
               AutoImageProcessor.from_pretrained('google/efficientnet-b2'); \
               AutoModelForImageClassification.from_pretrained('google/efficientnet-b2')"
# --- END FIX ---

# Copy the rest of the application code
COPY . .

# Expose the port
EXPOSE 8501

# Run the app
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]