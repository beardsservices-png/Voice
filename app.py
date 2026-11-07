"""My Voice: clone a voice from a short recording and make it say anything.

Runs on Railway (CPU). Everything worth keeping lives on the /data volume:
  /data/hf       the model download (~3 GB, fetched once)
  /data/voices   saved voice recordings, one .wav per name
Installable from Chrome (address bar -> Install). Protected by APP_PASSWORD.
"""
import os
import re
import shutil
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
os.environ.setdefault("HF_HOME", str(DATA_DIR / "hf"))  # must be set before the model library loads

import gradio as gr
import numpy as np
import torch
from chatterbox.tts import ChatterboxTTS

VOICES_DIR = DATA_DIR / "voices"
VOICES_DIR.mkdir(parents=True, exist_ok=True)
ICON = str(Path(__file__).parent / "static" / "icon.png")
MAX_CHARS = 3000
PASSWORD = os.environ.get("APP_PASSWORD", "")
USERNAME = os.environ.get("APP_USERNAME", "me")

if not PASSWORD:
    # A cloned voice open to anyone with the link could be made to say anything. Refuse.
    raise SystemExit("APP_PASSWORD is not set. Set it on the service and redeploy.")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _allowed_cpus() -> int:
    """CPUs this container may actually use. os.cpu_count() reports the whole host
    (48 on Railway), and that many threads fighting over a few cores is far slower."""
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)


torch.set_num_threads(int(os.environ.get("TORCH_THREADS", _allowed_cpus())))
print(f"Loading the voice model on {DEVICE} with {torch.get_num_threads()} threads "
      f"(first start downloads about 3 GB into {os.environ['HF_HOME']})...", flush=True)
MODEL = ChatterboxTTS.from_pretrained(device=DEVICE)
print("Ready.", flush=True)

_conditioned = {"key": None}


def _safe_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9 _-]", "", (name or "")).strip()
    return name[:40]


def saved_voices() -> list[str]:
    return sorted(p.stem for p in VOICES_DIR.glob("*.wav"))


def _sentences(text: str) -> list[str]:
    return [p for p in re.split(r"(?<=[.!?])\s+", " ".join(text.split())) if p]


def save_voice(recording, name):
    name = _safe_name(name)
    if not recording:
        raise gr.Error("Record or upload a voice first.")
    if not name:
        raise gr.Error("Give the voice a name (letters, numbers, spaces).")
    # Convert whatever was recorded/uploaded to a plain wav so it loads the same every time.
    import librosa
    import soundfile as sf
    audio, sr = librosa.load(recording, sr=None, mono=True)
    if len(audio) < sr * 5:
        raise gr.Error("That recording is under 5 seconds. 10-30 seconds of normal talking works best.")
    sf.write(VOICES_DIR / f"{name}.wav", audio, sr)
    gr.Info(f"Saved '{name}'.")
    return gr.update(choices=saved_voices(), value=name)


def delete_voice(name):
    if name and (VOICES_DIR / f"{name}.wav").is_file():
        (VOICES_DIR / f"{name}.wav").unlink()
        gr.Info(f"Deleted '{name}'.")
    voices = saved_voices()
    return gr.update(choices=voices, value=voices[0] if voices else None)


def speak(saved, recording, text, expressiveness, pace, progress=gr.Progress()):
    if recording:
        voice_path = recording
    elif saved and (VOICES_DIR / f"{saved}.wav").is_file():
        voice_path = str(VOICES_DIR / f"{saved}.wav")
    else:
        raise gr.Error("Pick a saved voice, or record one.")
    if not text or not text.strip():
        raise gr.Error("Type something for it to say.")
    if len(text) > MAX_CHARS:
        raise gr.Error(f"Keep it under {MAX_CHARS} characters at a time.")

    key = (voice_path, os.path.getmtime(voice_path), expressiveness)
    if key != _conditioned["key"]:
        MODEL.prepare_conditionals(voice_path, exaggeration=expressiveness)
        _conditioned["key"] = key

    sentences = _sentences(text)
    gap = np.zeros(int(MODEL.sr * 0.25), dtype=np.float32)
    pieces = []
    for i, sentence in enumerate(sentences):
        progress((i, len(sentences)), desc=f"Sentence {i + 1} of {len(sentences)}")
        with torch.inference_mode():
            wav = MODEL.generate(sentence, exaggeration=expressiveness, cfg_weight=pace)
        pieces += [wav.squeeze(0).cpu().numpy().astype(np.float32), gap]
    return MODEL.sr, np.concatenate(pieces)


with gr.Blocks(title="My Voice") as demo:
    gr.Markdown("# My Voice")
    with gr.Row():
        saved = gr.Dropdown(choices=saved_voices(), value=(saved_voices() or [None])[0],
                            label="Voice", scale=3)
        delete_btn = gr.Button("Delete", size="sm", scale=1)
    with gr.Accordion("Add a new voice", open=not saved_voices()):
        gr.Markdown("Talk normally for 10-30 seconds in a quiet room, or upload a recording. "
                    "You can use it right away, or save it to keep it.")
        recording = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Recording")
        with gr.Row():
            new_name = gr.Textbox(label="Name", placeholder="e.g. Me", scale=3)
            save_btn = gr.Button("Save voice", scale=1)
    text = gr.Textbox(lines=4, label="What should it say?", max_length=MAX_CHARS)
    with gr.Accordion("Fine tune", open=False):
        expressiveness = gr.Slider(0.25, 1.0, value=0.5, step=0.05, label="Expressiveness")
        pace = gr.Slider(0.2, 1.0, value=0.5, step=0.05, label="Pace (lower = slower, more deliberate)")
    speak_btn = gr.Button("Speak", variant="primary")
    gr.Markdown("A sentence takes several seconds here (no graphics card), so longer text takes a while.")
    out = gr.Audio(label="Result (download arrow saves it)", type="numpy")

    save_btn.click(save_voice, [recording, new_name], saved)
    delete_btn.click(delete_voice, saved, saved)
    speak_btn.click(speak, [saved, recording, text, expressiveness, pace], out)
    # Refresh the list each time the page opens, so a voice saved on another device shows up.
    demo.load(lambda: gr.update(choices=saved_voices(), value=(saved_voices() or [None])[0]), None, saved)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1, max_size=20).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        auth=(USERNAME, PASSWORD),
        auth_message="My Voice",
        pwa=True,
        favicon_path=ICON,
        show_error=True,
    )
