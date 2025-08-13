# backend/Dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Sources are relative to the build context (backend/)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Bring in the rest of the app from backend/
COPY . .

ENV XDG_CACHE_HOME=/tmp/cache TTS_HOME=/tmp/tts HF_HOME=/tmp/hf COQUI_TOS_AGREED=1
EXPOSE 10000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
