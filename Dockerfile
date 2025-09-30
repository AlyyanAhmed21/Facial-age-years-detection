# Start with the python-slim base image
FROM python:3.11-slim

# Set environment variables to prevent caching and optimize
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies and clean up in the same layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# --- THE FIX: Install packages and immediately clean up the cache ---
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip
# --- END FIX ---

# Copy the application code
COPY . .

# Expose the port
EXPOSE 8501

# Run the app
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]