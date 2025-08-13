# frontend/gradio_app.py
import os, requests, gradio as gr

# Prefer private-network host:port from Render; fall back to manual URL for local dev
BACKEND = (
    f"http://{os.getenv('BACKEND_HOSTPORT')}"
    if os.getenv("BACKEND_HOSTPORT")
    else os.getenv("BACKEND_URL", "http://localhost:9000")
)

def voice_turn(mic_audio):
    import soundfile as sf, io
    sr, y = mic_audio
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    r = requests.post(f"{BACKEND}/voice-turn", files={"audio": ("input.wav", buf, "audio/wav")}, stream=True)
    r.raise_for_status()
    reply_text = r.headers.get("X-Reply-Text", "")
    return (sr, y), reply_text, (22050, r.content)

with gr.Blocks(title="Armenian MH Voice Agent") as demo:
    gr.Markdown("## 🇦🇲 Զրուցակից՝ հոգեկան առողջության աջակցության համար *(փորձարարական)*")
    mic = gr.Audio(sources=["microphone"], type="numpy", label="Խոսեք և բաց թողեք…")
    reply_text = gr.Textbox(label="Պատասխան (տեքստ)")
    reply_audio = gr.Audio(label="Պատասխան (ձայն)", autoplay=True)
    gr.Button("Ուղարկել ձայնային շրջանը").click(voice_turn, inputs=mic, outputs=[mic, reply_text, reply_audio])

# Bind to the port Render assigns
port = int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "7860")))
demo.launch(server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"), server_port=port)
