FROM python:3.11-slim

# System deps needed by unstructured, pypdf, sentence-transformers
RUN apt-get update && apt-get install -y \
    libmagic1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer-cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# HF Spaces runs as a non-root user — make sure start.sh is executable
RUN chmod +x start.sh

# HF Spaces always exposes port 7860
EXPOSE 7860

CMD ["./start.sh"]
