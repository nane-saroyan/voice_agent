FROM python:3.11-slim

# System deps for audio + TTS
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY backend/ ./backend/

# Persist caches on Render disk (render.yaml mounts /opt/models)
ENV XDG_CACHE_HOME=/opt/models/cache \
    TTS_HOME=/opt/models/tts \
    HF_HOME=/opt/models/hf \
    COQUI_TOS_AGREED=1

# Command is overridden by render.yaml dockerCommand (uses $PORT)
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "10000"]
