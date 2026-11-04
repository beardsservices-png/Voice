FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 DATA_DIR=/data HF_HOME=/data/hf
# ffmpeg lets uploads in mp3/m4a/ogg be read, not just wav.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# CPU build of PyTorch (Railway has no GPU); much smaller than the default CUDA one.
RUN pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .
COPY static ./static

CMD ["python", "app.py"]
