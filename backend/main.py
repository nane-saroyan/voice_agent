import io
import os
import re
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse
from pydantic import BaseModel

# ===== Config via ENV =====
ASR_MODEL = os.getenv("ASR_MODEL", "small")  # "small","medium","large-v3", etc.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TTS_MODEL_NAME = os.getenv("TTS_MODEL_NAME", "tts_models/multilingual/multi-dataset/xtts_v2")
REFERENCE_SPEAKER_WAV = os.getenv("REFERENCE_SPEAKER_WAV", "")  # optional voice clone wav path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ASR_COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

# ===== App =====
app = FastAPI(title="Armenian Mental Health Voice Agent (Free, single service)")

ARMENIAN_SYSTEM_PROMPT = """Դու հոգեկան առողջության աջակցության զրուցակից ես՝ ոչ թե թերապևտ։
- Խոսիր պարզ, էմպատիկ հայերենով։
- Մի դարձրու ախտորոշում։ Մի տրամադրիր բժշկական ցուցումներ։
- Լսիր, ամփոփիր, հարցրու բաց հարցեր, առաջարկիր ինքնօգնության փոքր քայլեր (շնչառություն, գրանցում, քայլք)։
- Վթարային իրավիճակներում հստակ ասա՝ Հայաստանում զանգել **112**։
- Հիշեցրու, որ սա չի փոխարինում մասնագետին։
Պատասխանիր կարճ (1–3 պարբերություն)։"""

CRISIS_REGEX = re.compile(
    r"(ինքնասպան|սպանեմ ինձ|վնասեմ ինձ|չեմ ուզում ապրել|suicide|kill myself)",
    re.IGNORECASE
)

# ===== Models (lazy import/init to keep startup light on free) =====
_asr_model = None
_tts_model = None

def get_asr():
    global _asr_model
    if _asr_model is None:
        from faster_whisper import WhisperModel
        _asr_model = WhisperModel(ASR_MODEL, device=DEVICE, compute_type=ASR_COMPUTE_TYPE)
    return _asr_model

def get_tts():
    global _tts_model
    if _tts_model is None:
        from TTS.api import TTS
        _tts_model = TTS(TTS_MODEL_NAME).to(DEVICE)
    return _tts_model

# ===== Helpers =====
def detect_crisis(text: str) -> bool:
    return bool(CRISIS_REGEX.search(text or ""))

def llm_reply(user_text: str) -> str:
    # Crisis fast-path
    if detect_crisis(user_text):
        return (
            "Շատ ցավում եմ, որ հիմա այսքան ծանր ես զգում։ Քո անվտանգությունն առաջնահերթ է։ "
            "Խնդրում եմ անմիջապես դիմել շտապ օգնության՝ զանգելով **112** կամ դիմել մոտակա բժշկական կենտրոն։ "
            "Եթե կարող ես, տեղեկացրու վստահելի հարազատին/ընկերոջը, որ քո կողքին լինի հիմա։"
        )

    if LLM_PROVIDER.lower() == "openai" and OPENAI_API_KEY:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        msg = [
            {"role": "system", "content": ARMENIAN_SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=msg,
            temperature=0.6,
        )
        return (resp.choices[0].message.content or "").strip()

    # Local or fallback stub
    return "Շնորհակալ եմ, որ կիսվեցիր։ Կցանա՞ս պատմել, թե կոնկրետ ինչը քեզ ամենաշատն է անհանգստացնում հիմա։"

def wav_bytes_from_numpy(float_pcm: np.ndarray, sr: int = 22050) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, float_pcm, samplerate=sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()

# ===== Schemas =====
class ChatRequest(BaseModel):
    text: str
    session_id: Optional[str] = None

# ===== Routes (programmatic) =====
@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "ok"

@app.post("/asr")
async def transcribe(audio: UploadFile = File(...)):
    """
    Accept a wav/mp3/m4a file and return Armenian text.
    """
    import tempfile, os
    asr = get_asr()
    suffix = os.path.splitext(audio.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        data = await audio.read()
        tmp.write(data)
        tmp_path = tmp.name

    segments, info = asr.transcribe(tmp_path, language="hy", vad_filter=True)
    text = " ".join(seg.text for seg in segments).strip()
    os.remove(tmp_path)
    return {"text": text, "language": info.language}

@app.post("/chat")
async def chat(req: ChatRequest):
    reply = llm_reply(req.text)
    return {"reply": reply}

@app.post("/tts")
async def synthesize(text: str = Form(...)):
    tts = get_tts()
    wav = tts.tts(text=text, speaker_wav=REFERENCE_SPEAKER_WAV or None, language="hy")
    audio = wav_bytes_from_numpy(wav, sr=22050)
    headers = {"Content-Disposition": f'inline; filename="reply_{datetime.utcnow().isoformat()}.wav"'}
    return StreamingResponse(io.BytesIO(audio), media_type="audio/wav", headers=headers)

@app.post("/voice-turn")
async def voice_turn(audio: UploadFile = File(...)):
    """
    One-shot: audio -> text -> safe reply -> TTS -> stream WAV
    Also echoes transcript & reply text via headers.
    """
    asr = get_asr()
    tts = get_tts()

    import tempfile, os
    suffix = os.path.splitext(audio.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        data = await audio.read()
        tmp.write(data)
        tmp_path = tmp.name

    segments, info = asr.transcribe(tmp_path, language="hy", vad_filter=True)
    user_text = " ".join(seg.text for seg in segments).strip()
    os.remove(tmp_path)

    reply_text = llm_reply(user_text)
    wav = tts.tts(text=reply_text, speaker_wav=REFERENCE_SPEAKER_WAV or None, language="hy")
    audio = wav_bytes_from_numpy(wav, sr=22050)

    headers = {
        "X-Transcript": user_text.encode("utf-8").decode("utf-8"),
        "X-Reply-Text": reply_text.encode("utf-8").decode("utf-8"),
    }
    return StreamingResponse(io.BytesIO(audio), media_type="audio/wav", headers=headers)

# ===== Gradio UI mounted on the same app (free-tier friendly) =====
import gradio as gr
import tempfile

def _silence(sr=22050, sec=0.5) -> Tuple[int, np.ndarray]:
    n = int(sr * sec)
    return sr, np.zeros(n, dtype=np.float32)

def _gradio_turn(mic_audio: Tuple[int, np.ndarray]):
    """
    ASR → LLM → TTS
    Returns:
      - reply text (str)
      - reply audio (sr:int, np.ndarray)  # NEVER None/bool/raw-bytes
    """
    try:
        # Validate input
        if not mic_audio or not isinstance(mic_audio, tuple) or len(mic_audio) != 2:
            return "Չկա միկրոֆոնի ձայնագրություն։", _silence()
        sr, y = mic_audio
        if y is None or isinstance(y, bool):
            return "Չկա միկրոֆոնի ձայնագրություն։", _silence()

        # Save a temporary WAV for Whisper
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp, y, int(sr), format="WAV", subtype="PCM_16")
            tmp_path = tmp.name

        # ASR
        asr = get_asr()
        segments, _info = asr.transcribe(tmp_path, language="hy", vad_filter=True)
        user_text = " ".join(seg.text for seg in segments).strip()

        # LLM
        reply_text = llm_reply(user_text)

        # TTS -> numpy waveform
        tts = get_tts()
        wav_np = tts.tts(text=reply_text, speaker_wav=REFERENCE_SPEAKER_WAV or None, language="hy")

        # Coerce to the exact dtype/shape Gradio expects
        if isinstance(wav_np, (list, tuple)):
            wav_np = np.asarray(wav_np, dtype=np.float32)
        elif isinstance(wav_np, np.ndarray):
            wav_np = wav_np.astype(np.float32, copy=False)
        else:
            # Unexpected type (e.g., num/str/bool) -> return silence
            return f"Սխալ տեղի ունեցավ․ սխալ ձայնային ֆորմատ ({type(wav_np).__name__})", _silence()

        return reply_text, (22050, wav_np)

    except Exception as e:
        # Return readable message + valid audio tuple (silence) to keep Gradio happy
        return f"Սխալ տեղի ունեցավ․ {type(e).__name__}: {e}", _silence()

with gr.Blocks(title="🇦🇲 Զրուցակից՝ հոգեկան առողջության աջակցման համար") as demo:
    gr.Markdown(
        "### Բարև․ սա փորձարարական ձայնային զրուցակից է հոգեկան բարեկեցության համար.\n"
        "**Չի փոխարինում մասնագետին**։ Վթարային իրավիճակում զանգահարեք **112**։"
    )
    mic = gr.Audio(sources=["microphone"], type="numpy", label="Խոսեք և բաց թողեք…")
    reply_text = gr.Textbox(label="Պատասխան (տեքստ)")
    reply_audio = gr.Audio(label="Պատասխան (ձայն)", autoplay=True)

    # Only two outputs: text + audio. Do NOT return the mic component itself.
    gr.Button("Ուղարկել ձայնային շրջանը").click(
        _gradio_turn,
        inputs=mic,
        outputs=[reply_text, reply_audio],
    )

# Mount UI at "/"
app = gr.mount_gradio_app(app, demo, path="/")
