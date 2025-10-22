# ===========================
# 🧠 Visora AI UCVE-X Dockerfile (GPU + CPU auto)
# ===========================

FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y \
    git ffmpeg libsm6 libxext6 libgl1-mesa-glx curl wget python3-dev build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && pip install \
    fastapi uvicorn numpy opencv-python pillow moviepy pydub textblob firebase-admin redis rq prometheus-client \
    torch torchvision torchaudio diffusers transformers accelerate requests gTTS pyttsx3 stripe razorpay

ENV PYTHONUNBUFFERED=1
ENV RENDER_WATERMARK="Visora AI"
ENV TTS_PROVIDER="elevenlabs"

EXPOSE 8000

CMD ["uvicorn", "visora_ai_ucve_x_final:app", "--host", "0.0.0.0", "--port", "8000"]
