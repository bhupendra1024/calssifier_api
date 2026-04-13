FROM python:3.12-slim

# Set working directory
WORKDIR /app

# 1. Install system dependencies (git is needed for some open_clip installs)
RUN apt-get update && apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# 2. Install Python dependencies
# Done before copying code to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. PRE-DOWNLOAD MODEL WEIGHTS
# We set the cache directory environment variable so both the build
# and the runtime app look in the same place.
ENV HF_HOME=/app/model_cache
ENV TORCH_HOME=/app/model_cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/model_cache

COPY download_models.py .
RUN python download_models.py

# 4. Copy the application code
# We do this LAST so that code changes don't trigger a model re-download
# .dockerignore excludes .venv, .git, __pycache__, .env, model_cache, etc.
COPY . .

# 5. Environment settings
ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# Entry point
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
