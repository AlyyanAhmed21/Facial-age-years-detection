# Start with the python-slim base image
FROM python:3.11-slim

# --- THE DEFINITIVE FIX: Install ALL known dependencies for OpenCV ---
# This is the "kitchen sink" approach to ensure everything is included.
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy and install Python requirements
# We run as root to simplify paths and permissions, which is acceptable for this single-app container
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the port
EXPOSE 8501

# Run the app. Using `python -m` is the most robust way.
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]