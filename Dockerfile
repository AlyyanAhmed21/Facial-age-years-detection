FROM python:3.11-slim

# Install OS dependencies required by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Create a non-root user for security
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app

# Copy requirements file first to leverage Docker's cache
COPY --chown=user:user requirements.txt .

# Install dependencies, including the CPU version of PyTorch
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY --chown=user:user . .

# Tell Docker that the container listens on port 8501
EXPOSE 8501

# Command to run the Streamlit app when the container starts
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
