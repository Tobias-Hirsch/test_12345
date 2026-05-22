# Developer: Jinglu Han
# mailbox: admin@de-manufacturing.cn

# Use an official PaddlePaddle image as a parent image
# This image comes with Python, CUDA, and cuDNN pre-installed
FROM paddlepaddle/paddle:3.1.0-gpu-cuda12.6-cudnn9.5

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies using a trusted mirror
# We need paddleocr, and a web framework (fastapi) to serve the model
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip uninstall -y paddleocr paddlex paddlex-hpi langchain langchain-core langchain-community || true && \
    pip install --no-cache-dir \
        paddleocr==3.1.0 \
        paddlex==3.1.0 \
        langchain==0.2.17 \
        fastapi \
        uvicorn \
        python-multipart \
        Pillow \
        opencv-python-headless

# Copy the application code into the container
COPY ./main.py /app/main.py

# Expose the port the app runs on
EXPOSE 8080

# Define the command to run the application
# The --workers flag can be adjusted based on the server's CPU cores
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]