# Use a standard Python 3.11 base image
FROM python:3.11-slim

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
# The --server.address=0.0.0.0 is crucial to make it accessible from outside
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]