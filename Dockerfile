FROM python:3.11-slim

# --- THE DEFINITIVE FIX FOR OPENCV ---
# Install the correct, complete set of headless dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
# --- END FIX ---

# Create a non-root user for security
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app

# Copy and install Python requirements
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY --chown=user:user . .

# Expose the port
EXPOSE 8501

# Use your robust CMD instruction to run the app
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]