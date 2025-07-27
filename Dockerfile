FROM python:3.10-slim

# Avoid interactive prompts and speed up build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies required by Tesseract, Pillow, etc.
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy the rest of the project files
COPY . .

# Create output directory if not present
RUN mkdir -p ./sample_dataset/output

# Run the processing script
CMD ["python", "process_pdfs.py"]
