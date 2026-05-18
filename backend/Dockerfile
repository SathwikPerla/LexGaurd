FROM python:3.11-slim-bookworm

WORKDIR /app

# build-essential: compile Python C extensions (scikit-learn, etc.)
# libglib2.0-0:   runtime dep for PyMuPDF on Linux
# libgl1-mesa-glx was removed — renamed to libgl1 in bookworm, not needed by our stack
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source (dockerignore excludes .env, .venv, tests, etc.)
COPY . .

# Cloud Run injects PORT; default 8080
ENV PORT=8080

EXPOSE 8080

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}
