# Armenian Mental Health Voice Agent (FastAPI + Gradio) — Render Free

Single service app (FastAPI backend + Gradio UI) deployable on Render **Free** plan.

## Features
- Armenian speech → text (Whisper via faster-whisper)
- Supportive, safety-aware LLM replies (OpenAI or local stub)
- Armenian text → speech (Coqui TTS XTTS v2)
- Crisis detection (regex) → immediate emergency guidance (dial **112** in Armenia)
- Gradio UI mounted at `/` on the same FastAPI app
- Healthcheck at `/healthz`

---

## Deploy on Render (Free)

1. Fork/clone this repo.
2. In Render: **New → Blueprint** and pick your repo.
3. When prompted, set:
   - `OPENAI_API_KEY` (if using OpenAI)
   - Optional: adjust `ASR_MODEL` (`small` is OK for free CPU), `TTS_MODEL_NAME`
4. Deploy. When it’s live, open the service URL — you should see the Gradio app.

> Note: On the free plan, model downloads (Whisper/XTTS) happen on first run and on each cold start. Expect a slow first interaction.

---

## Local dev

```bash
cd backend
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run
uvicorn main:app --host 0.0.0.0 --port 7860
# Open http://localhost:7860
