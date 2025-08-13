FROM python:3.11-slim

# System deps for audio encode/decode and Coqui TTS
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy app code (dockerContext is backend/, so "." is the backend folder)
COPY . .

# Helpful caches (won't persist on free plan, but fine)
ENV XDG_CACHE_HOME=/tmp/cache \
    TTS_HOME=/tmp/tts \
    HF_HOME=/tmp/hf \
    COQUI_TOS_AGREED=1

EXPOSE 10000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
