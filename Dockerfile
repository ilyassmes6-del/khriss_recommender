FROM python:3.11-slim

# System deps: libgl/glib for pillow/torchvision image ops.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU torch wheels + the rest.
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Pre-download the OpenCLIP weights into the image so first request is warm.
RUN python -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')"

COPY app ./app
COPY indexer.py ./indexer.py

EXPOSE 8000

# Shell form on purpose: managed hosts (Railway, Render, Cloud Run) assign the
# port at runtime via $PORT. Falls back to 8000 for local compose.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
