# Armenian Mental Health Voice Agent (FastAPI + Gradio) — Render Free

Single service app (FastAPI backend + Gradio UI) deployable on Render **Free** plan.

## Features
- Armenian speech → text (Whisper via faster-whisper)
- Supportive, safety-aware LLM replies (OpenAI or local stub)
- Armenian text → speech (Coqui TTS XTTS v2)
- Crisis detection (regex) → immediate emergency guidance (dial **112** in Armenia)
- Gradio UI mounted at `/` on the same FastAPI app
- Healthcheck at `/healthz`
