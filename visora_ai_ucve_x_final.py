#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visora_ai_ucve_x_final.py
Visora AI UCVE-X (hybrid GPU/CPU compatible, v31 merged)
Features:
 - GPU autodetect (torch.cuda.is_available)
 - ElevenLabs TTS primary, gTTS/pyttsx3 fallback
 - Multi-language support for India (user-selectable)
 - Prompt optimizer, render endpoints, advanced v31 pipeline
 - SFX, lipsync, temporal stubs, upscale stub
 - Payment & Firebase hooks (stubs)
 - Single-file drop-in for Render or local
Author: prepared for you
"""
import os
import re
import uuid
import json
import logging
import time
from datetime import datetime
from typing import List, Optional, Dict
from fastapi import FastAPI, Form, UploadFile, File, BackgroundTasks, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect

# Initialize app
app = FastAPI()

# ✅ Enable CORS & WS connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------- Optional imports (graceful) ----------
PIL_AVAILABLE = False
MOVIEPY_AVAILABLE = False
TORCH_AVAILABLE = False
DIFFUSERS_AVAILABLE = False
REQUESTS_AVAILABLE = False
PYDUB_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except Exception as e:
    logging.info("PIL not available: %s", e)

try:
    from moviepy.editor import ImageSequenceClip, AudioFileClip
    MOVIEPY_AVAILABLE = True
except Exception as e:
    logging.info("moviepy not available: %s", e)

try:
    import torch
    TORCH_AVAILABLE = True
except Exception as e:
    logging.info("torch not available: %s", e)

try:
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    DIFFUSERS_AVAILABLE = True
except Exception as e:
    logging.info("diffusers not available: %s", e)

try:
    import requests
    REQUESTS_AVAILABLE = True
except Exception as e:
    logging.info("requests not available: %s", e)

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except Exception as e:
    logging.info("pydub not available: %s", e)

# TTS optional libs (gTTS/pyttsx3)
GTTs_AVAILABLE = False
PYTTSX3_AVAILABLE = False
try:
    from gtts import gTTS
    GTTs_AVAILABLE = True
except Exception:
    GTTs_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except Exception:
    PYTTSX3_AVAILABLE = False

# ---------- App init & config ----------
app = FastAPI(title="Visora AI UCVE-X", version="v31")

DOMAIN = os.getenv("DOMAIN", "http://127.0.0.1:8000")
HUGGINGFACE_HUB_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
ELEVEN_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVEN_VOICE_ID = os.getenv("ELEVEN_VOICE_ID", None)
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "elevenlabs")  # elevenlabs / gtts / pyttsx3
RENDER_WATERMARK = os.getenv("RENDER_WATERMARK", "Visora AI")
DEFAULT_STYLE = os.getenv("DEFAULT_STYLE", "realistic")

# Supported Indian languages (language code for gTTS / hint for ElevenLabs)
SUPPORTED_LANGUAGES = {
    "hi": "Hindi",
    "en": "English",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "or": "Odia",
    "as": "Assamese"
}

# ---------------- Utility helpers ----------------
def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def save_upload_file(upload: UploadFile, prefix="upload") -> str:
    ensure_dir("/tmp")
    path = f"/tmp/{prefix}_{uuid.uuid4().hex[:8]}_{upload.filename}"
    with open(path, "wb") as fh:
        fh.write(upload.file.read())
    return path

def _watermark_image_pil(img_path: str, out_path: str, text: str = RENDER_WATERMARK):
    if not PIL_AVAILABLE:
        return img_path
    try:
        im = Image.open(img_path).convert("RGBA")
        draw = ImageDraw.Draw(im)
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except Exception:
            font = ImageFont.load_default()
        w, h = im.size
        margin = 8
        tw, th = draw.textsize(text, font=font)
        pos = (w - tw - margin, h - th - margin)
        draw.text(pos, text, fill=(255,255,255,200), font=font)
        im.save(out_path)
        return out_path
    except Exception:
        return img_path

# Torch device helper
def _get_device():
    if TORCH_AVAILABLE and torch.cuda.is_available():
        logging.info("Using CUDA device")
        return "cuda"
    logging.info("Using CPU device")
    return "cpu"

# Simple job checkpoint
JOBS_DIR = "/tmp/visora_jobs"
ensure_dir(JOBS_DIR)
def save_job(job: dict):
    path = os.path.join(JOBS_DIR, f"{job['id']}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(job, fh, ensure_ascii=False, indent=2)
def load_job(job_id: str) -> Optional[dict]:
    path = os.path.join(JOBS_DIR, f"{job_id}.json")
    if os.path.exists(path):
        return json.load(open(path, "r", encoding="utf-8"))
    return None

# ---------------- Prompt optimizer ----------------
def prompt_optimizer_simple(prompt: str, style: str = DEFAULT_STYLE) -> str:
    templates = {
        "cartoon": "A bright, colorful cartoon illustration of {p}, cel-shaded, characterful",
        "realistic": "A photorealistic cinematic scene of {p}, ultra-detailed textures, cinematic lighting, shallow depth of field",
        "anime": "An anime scene of {p}, clean lines, expressive lighting, soft gradients",
        "cinematic": "A cinematic film-frame of {p}, wide-angle, dramatic lighting, film grain"
    }
    tpl = templates.get(style, templates["realistic"])
    expanded = tpl.format(p=prompt.strip()) + ", ultra detailed, high quality"
    return expanded

# ---------------- TTS: ElevenLabs + fallback ----------------
def _elevenlabs_synthesize(text: str, voice_id: Optional[str]=None, out_path: str="/tmp/visora_eleven_tts.mp3"):
    if not REQUESTS_AVAILABLE or not ELEVEN_API_KEY:
        return None
    vid = voice_id or ELEVEN_VOICE_ID
    if not vid:
        # try to fetch voice list and pick first
        try:
            r = requests.get("https://api.elevenlabs.io/v1/voices", headers={"xi-api-key": ELEVEN_API_KEY}, timeout=15)
            if r.status_code == 200:
                data = r.json()
                voices = data.get("voices", [])
                if voices:
                    vid = voices[0].get("voice_id") or voices[0].get("id")
        except Exception:
            vid = None
    if not vid:
        return None
    try:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{vid}"
        headers = {"xi-api-key": ELEVEN_API_KEY, "Content-Type": "application/json"}
        payload = {"text": text}
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code == 200:
            with open(out_path, "wb") as fh:
                fh.write(r.content)
            return out_path
        else:
            logging.error("ElevenLabs TTS failed: %s %s", r.status_code, r.text)
            return None
    except Exception:
        logging.exception("ElevenLabs synth error")
        return None

def _generate_tts(text: str, lang: str = "hi", out_path: str = "/tmp/visora_tts.mp3"):
    # prefer ElevenLabs (if configured)
    if TTS_PROVIDER == "elevenlabs" and ELEVEN_API_KEY:
        out = _elevenlabs_synthesize(text, out_path=out_path)
        if out:
            return out
    # fallback gTTS
    if GTTs_AVAILABLE:
        try:
            tts = gTTS(text=text, lang=lang)
            tts.save(out_path)
            return out_path
        except Exception:
            pass
    # fallback pyttsx3 offline
    if PYTTSX3_AVAILABLE:
        try:
            engine = pyttsx3.init()
            engine.save_to_file(text, out_path)
            engine.runAndWait()
            return out_path
        except Exception:
            pass
    return None

# ---------------- Diffusers frame generator (starter) ----------------
def _load_sd_pipeline(model_id: str = "runwayml/stable-diffusion-v1-5"):
    if not DIFFUSERS_AVAILABLE or not TORCH_AVAILABLE:
        raise RuntimeError("diffusers/torch not installed")
    device = _get_device()
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype, use_auth_token=HUGGINGFACE_HUB_TOKEN)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    pipe = pipe.to(device)
    return pipe

def _synthesize_frames(pipe, prompt: str, num_frames: int=24, width: int=512, height: int=512, guidance_scale: float=7.5, seed: Optional[int]=None):
    out = []
    try:
        device = _get_device()
        gen = torch.Generator(device=device if device=="cuda" else "cpu")
        gen = gen.manual_seed(seed or int.from_bytes(os.urandom(2), "big"))
        for i in range(num_frames):
            frame_prompt = f"{prompt}, cinematic, frame {i}"
            with torch.autocast(device) if (TORCH_AVAILABLE and device=="cuda") else torch.cpu.amp.autocast(enabled=False):
                res = pipe(frame_prompt, height=height, width=width, guidance_scale=guidance_scale, generator=gen)
                img = res.images[0]
            p = f"/tmp/visora_frame_{uuid.uuid4().hex[:6]}_{i}.png"
            img.save(p)
            wm = p.replace(".png", "_wm.png")
            _watermark_image_pil(p, wm)
            out.append(wm)
    except Exception as e:
        logging.exception("Frame synthesis failed")
    return out

# ---------------- Temporal smoothing (stub) ----------------
def temporal_smooth_frames(frames: List[str]) -> List[str]:
    # placeholder: return same frames; in production integrate optical-flow / temporal models
    return frames

# ---------------- SFX selection & mix ----------------
SFX_BANK = {
    "running": None,
    "rain": None,
    "explosion": None,
    "wind": None,
    "default": None
}

def select_sfx(prompt: str) -> Optional[str]:
    p = prompt.lower()
    if "run" in p or "running" in p: return SFX_BANK.get("running")
    if "rain" in p or "storm" in p: return SFX_BANK.get("rain")
    if "explode" in p or "bomb" in p: return SFX_BANK.get("explosion")
    return SFX_BANK.get("default")

def mix_audio_tracks(tts_path: Optional[str], sfx_path: Optional[str], duration_sec: int, out_path: str):
    if not PYDUB_AVAILABLE:
        return tts_path
    try:
        target_ms = int(duration_sec*1000)
        if sfx_path and os.path.exists(sfx_path):
            bg = AudioSegment.from_file(sfx_path)
            bg_loop = bg * (target_ms // len(bg) + 1) if len(bg)>0 else bg
            bg_clip = bg_loop[:target_ms]
            if tts_path and os.path.exists(tts_path):
                tts = AudioSegment.from_file(tts_path)[:target_ms]
                mixed = bg_clip.overlay(tts - 6)
            else:
                mixed = bg_clip
            mixed.export(out_path, format="mp3")
            return out_path
        else:
            return tts_path
    except Exception:
        logging.exception("mix audio failed")
        return tts_path

# ---------------- Lipsync stub ----------------
def lipsync_stub(audio_path: str, reference_frame: Optional[str]=None):
    # In production replace with Wav2Lip integration (ethical use only)
    return audio_path

# ---------------- Upscale stub ----------------
def upscale_stub(video_path: str, scale: int=2) -> str:
    # placeholder returns same path; implement Real-ESRGAN pipeline to upscale
    out = video_path.replace(".mp4", f"_up{scale}x.mp4")
    try:
        shutil.copy(video_path, out)
        return out
    except Exception:
        return video_path

# ---------------- Memory store ----------------
MEMORY_FILE = "/tmp/visora_memory.json"
def _load_memory():
    if os.path.exists(MEMORY_FILE):
        return json.load(open(MEMORY_FILE, "r", encoding="utf-8"))
    return {"history":[]}
def _save_memory(mem: dict):
    with open(MEMORY_FILE, "w", encoding="utf-8") as fh:
        json.dump(mem, fh, ensure_ascii=False, indent=2)

# ---------------- API Endpoints ----------------

@app.get("/")
def root():
    return {"message": "Visora AI UCVE-X running", "time": datetime.utcnow().isoformat()}

@app.get("/health")
def health():
    return {"status":"ok", "time": datetime.utcnow().isoformat()}

@app.get("/languages")
def languages():
    """Return supported Indian languages"""
    return {"supported_languages": SUPPORTED_LANGUAGES}

# Simple render request (starter) - quick path
@app.post("/pro/render_request")
def pro_render_request(script_text: str = Form(...),
                       style: str = Form(DEFAULT_STYLE),
                       duration_sec: int = Form(6),
                       fps: int = Form(12),
                       width: int = Form(512),
                       height: int = Form(512),
                       lang: str = Form("hi"),
                       tts_text: Optional[str] = Form(None),
                       background_tasks: BackgroundTasks = None):
    """
    Basic render request:
      - optimize prompt
      - generate frames (diffusers if available else placeholder)
      - generate TTS in requested lang
      - mix SFX
      - compose video (MoviePy)
    """
  # APPLY PHOTO OVERLAYS IF MAPPING EXISTS
    try:
        _apply_character_photos_to_video(job_id, job.get("mapping", {}))
        # then apply motion
        apply_motion_to_dialogues(job_id, job.get("mapping", {}))
    except Exception as e:
        logging.info(f"No mapping/photos to apply: {e}")

    job_id = f"job_{uuid.uuid4().hex[:8]}"
    job = {"id": job_id, "status": "queued", "script": script_text, "style": style,
           "duration_sec": duration_sec, "fps": fps, "width": width, "height": height, "lang": lang}
    save_job(job)
    # process inline or background
    if background_tasks:
        background_tasks.add_task(_render_worker, job_id, tts_text)
        job["status"]="background_started"
    else:
        _render_worker(job_id, tts_text)
    save_job(job)
    return {"job_id": job_id, "status": job["status"], "message": "queued/started"}

def _render_worker(job_id: str, tts_text: Optional[str]=None):
    job = load_job(job_id)
    if not job:
        return
    try:
        job["status"]="running"
        save_job(job)
        optimized = prompt_optimizer_simple(job["script"], style=job.get("style", DEFAULT_STYLE))
        job["optimized_prompt"] = optimized
        save_job(job)
        # frames
        frames = []
        if DIFFUSERS_AVAILABLE and TORCH_AVAILABLE:
            pipe = _load_sd_pipeline()
            frames = _synthesize_frames(pipe, optimized, num_frames=max(1, job["duration_sec"]*job["fps"]), width=job["width"], height=job["height"])
        else:
            # create placeholder single frame and duplicate
            p = f"/tmp/visora_placeholder_{uuid.uuid4().hex[:6]}.png"
            if PIL_AVAILABLE:
                im = Image.new("RGB", (job["width"], job["height"]), (40,40,60))
                ImageDraw.Draw(im).text((20,20), optimized[:200], fill=(255,255,255))
                im.save(p)
            frames = [p] * max(1, job["duration_sec"]*job["fps"])
        job["frames"] = frames
        save_job(job)
        # temporal smoothing
        frames = temporal_smooth_frames(frames)
        job["frames_smoothed"] = frames
        save_job(job)
        # TTS
        audio_path = None
        if tts_text:
            lang = job.get("lang","hi")
            audio_path = _generate_tts(tts_text, lang=lang, out_path=f"/tmp/visora_tts_{uuid.uuid4().hex[:6]}.mp3")
            job["tts"] = audio_path
            save_job(job)
        # SFX mix
        sfx = select_sfx(job["script"])
        mixed = mix_audio_tracks(audio_path, sfx, job["duration_sec"], out_path=f"/tmp/visora_mix_{uuid.uuid4().hex[:6]}.mp3")
        job["mixed_audio"] = mixed
        save_job(job)
        # lipsync stub
        if mixed:
            job["lipsync"] = lipsync_stub(mixed, reference_frame=frames[0] if frames else None)
            save_job(job)
        # compose video
        video_out = f"/tmp/visora_video_{job_id}.mp4"
        if MOVIEPY_AVAILABLE and frames:
            clip = ImageSequenceClip(frames, fps=job["fps"])
            if mixed and os.path.exists(mixed):
                clip = clip.set_audio(AudioFileClip(mixed))
            clip.write_videofile(video_out, codec="libx264", audio_codec="aac", verbose=False, logger=None)
        else:
            # fallback: return first frame path
            video_out = frames[0] if frames else None
        job["video"] = video_out
        job["status"] = "done"
        save_job(job)
        # memory save
        mem = _load_memory()
        mem["history"].append({"time": datetime.utcnow().isoformat(), "script": job["script"], "style": job["style"]})
        _save_memory(mem)
    except Exception as e:
        logging.exception("render worker failed")
        job["status"] = "failed"
        job["error"] = str(e)
        save_job(job)

# ---------------- Advanced v31 endpoint ----------------
@app.post("/pro/v31_render_advanced")
def pro_v31_render_advanced(script_text: str = Form(...),
                            style: str = Form(DEFAULT_STYLE),
                            duration_sec: int = Form(6),
                            fps: int = Form(12),
                            width: int = Form(512),
                            height: int = Form(512),
                            lang: str = Form("hi"),
                            tts_text: Optional[str] = Form(None),
                            upscale: bool = Form(False),
                            prosody_pitch: float = Form(1.0),
                            prosody_rate: float = Form(1.0),
                            background_tasks: BackgroundTasks = None):
    """
    Full v31 advanced pipeline:
     - optimized prompt
     - animate-diffusion (if available) -> frames
     - temporal smoothing
     - advanced prosody TTS
     - SFX + mix
     - lipsync stub
     - compose video
     - optional upscale stub
    """
    job_id = f"v31_{uuid.uuid4().hex[:8]}"
    job = {"id": job_id, "status": "queued", "script": script_text, "style": style, "duration_sec": duration_sec, "fps": fps, "width": width, "height": height, "lang": lang}
    save_job(job)
    if background_tasks:
        background_tasks.add_task(_v31_worker, job_id, tts_text, upscale, prosody_pitch, prosody_rate)
        job["status"]="background_started"
    else:
        _v31_worker(job_id, tts_text, upscale, prosody_pitch, prosody_rate)
    save_job(job)
    return {"job_id": job_id, "status": job["status"], "message": "v31 queued/started"}

def _v31_worker(job_id: str, tts_text: Optional[str], upscale: bool, prosody_pitch: float, prosody_rate: float):
    job = load_job(job_id)
    if not job:
        return
    try:
        job["status"] = "running"
        save_job(job)
        optimized = prompt_optimizer_simple(job["script"], style=job["style"])
        job["optimized_prompt"] = optimized
        save_job(job)
        # text->motion frames: use SD pipeline as placeholder (or AnimateDiff if integrated)
        frames = []
        if DIFFUSERS_AVAILABLE and TORCH_AVAILABLE:
            try:
                pipe = _load_sd_pipeline()
                frames = _synthesize_frames(pipe, optimized, num_frames=max(1, job["duration_sec"]*job["fps"]), width=job["width"], height=job["height"])
            except Exception:
                frames = []
        if not frames:
            p = f"/tmp/visora_v31_placeholder_{uuid.uuid4().hex[:6]}.png"
            if PIL_AVAILABLE:
                im = Image.new("RGB", (job["width"], job["height"]), (30,30,40))
                ImageDraw.Draw(im).text((20,20), optimized[:200], fill=(255,255,255))
                im.save(p)
            frames = [p] * max(1, job["duration_sec"]*job["fps"])
        job["frames"] = frames
        save_job(job)
        # temporal smoothing
        frames = temporal_smooth_frames(frames)
        job["frames_smoothed"] = frames
        save_job(job)
        # advanced prosody TTS (here basic wrapper)
        audio_path = None
        if tts_text:
            # try ElevenLabs then fallback
            audio_path = _generate_tts(tts_text, lang=job.get("lang","hi"), out_path=f"/tmp/v31_tts_{uuid.uuid4().hex[:6]}.mp3")
            job["tts"] = audio_path
            save_job(job)
        # SFX
        sfx = select_sfx(job["script"])
        mixed = mix_audio_tracks(audio_path, sfx, job["duration_sec"], out_path=f"/tmp/v31_mix_{uuid.uuid4().hex[:6]}.mp3")
        job["mixed_audio"] = mixed
        save_job(job)
        # lipsync stub
        if mixed:
            job["lipsync_audio"] = lipsync_stub(mixed, reference_frame=frames[0] if frames else None)
            save_job(job)
        # compose
        video_out = f"/tmp/v31_video_{job_id}.mp4"
        if MOVIEPY_AVAILABLE and frames:
            clip = ImageSequenceClip(frames, fps=job["fps"])
            if mixed and os.path.exists(mixed):
                clip = clip.set_audio(AudioFileClip(mixed))
            clip.write_videofile(video_out, codec="libx264", audio_codec="aac", verbose=False, logger=None)
        else:
            video_out = frames[0] if frames else None
        job["video"] = video_out
        save_job(job)
        # upscale optional (stub)
        if upscale and video_out:
            job["video_upscaled"] = upscale_stub(video_out, scale=2)
            save_job(job)
        job["status"] = "done"
        save_job(job)
        mem = _load_memory()
        mem["history"].append({"time": datetime.utcnow().isoformat(), "script": job["script"], "style": job["style"]})
        _save_memory(mem)
    except Exception as e:
        logging.exception("v31 worker failed")
        job["status"] = "failed"
        job["error"] = str(e)
        save_job(job)

# ---------------- Job status endpoints ----------------
@app.get("/pro/job_status/{job_id}")
def pro_job_status(job_id: str):
    job = load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.post("/pro/cancel/{job_id}")
def pro_cancel(job_id: str):
    job = load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job["status"] = "cancelled"
    save_job(job)
    return {"status":"cancelled"}

# ---------------- Admin ----------------
@app.get("/admin/get_memory")
def admin_get_memory():
    return _load_memory()

# ---------------- Startup logs ----------------
@app.on_event("startup")
def startup_event():
    logging.info("Visora AI UCVE-X starting...")
    if TORCH_AVAILABLE:
        logging.info("Torch available. device=%s", ("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        logging.info("Torch not available; running CPU-only fallback")
    if ELEVEN_API_KEY:
        logging.info("ElevenLabs configured")
    logging.info("Supported languages: %s", list(SUPPORTED_LANGUAGES.keys()))

# ======================================================
# ⚡ Real-Time WebSocket + Firebase Sync Integration
# ======================================================
from fastapi import WebSocket, WebSocketDisconnect
import asyncio

active_connections = {}

@app.websocket("/ws/status/{job_id}")
async def websocket_status(websocket: WebSocket, job_id: str):
    await websocket.accept()
    active_connections[job_id] = websocket
    logging.info(f"Client connected for job {job_id}")

    try:
        progress = 0
        while True:
            # Simulate progress (replace with actual redis/fb data)
            progress_data = get_render_progress(job_id)
            await websocket.send_json(progress_data)
            
            if progress_data.get("status") == "completed":
                logging.info(f"Job {job_id} completed, closing socket.")
                await websocket.close()
                break

            await asyncio.sleep(5)
    except WebSocketDisconnect:
        logging.info(f"Client disconnected from job {job_id}")
        if job_id in active_connections:
            del active_connections[job_id]
    except Exception as e:
        logging.error(f"WebSocket error for job {job_id}: {str(e)}")
        if job_id in active_connections:
            del active_connections[job_id]


# 🧠 Dummy progress generator (replace with Redis/Firebase sync)
def get_render_progress(job_id: str):
    # Example structure — replace with actual logic
    return {
        "job_id": job_id,
        "progress": 78,  # 0–100
        "status": "rendering",
        "eta_sec": 42,
        "message": "Generating realistic frames..."
    }

# ======================================================
# 🎞️ Auto Thumbnail Preview + Live Frame Stream
# ======================================================
import io, base64
from moviepy.editor import VideoFileClip
from PIL import Image

@app.websocket("/ws/preview/{job_id}")
async def websocket_preview(websocket: WebSocket, job_id: str):
    await websocket.accept()
    logging.info(f"Live preview socket opened for job {job_id}")
    try:
        while True:
            frame_data = generate_preview_frame(job_id)
            if frame_data:
                await websocket.send_json(frame_data)
            await asyncio.sleep(10)  # every 10 sec
    except WebSocketDisconnect:
        logging.info(f"Preview disconnected for {job_id}")
    except Exception as e:
        logging.error(f"Preview error for {job_id}: {str(e)}")


def generate_preview_frame(job_id: str):
    """Extracts a single frame, encodes to base64 and uploads to Firebase"""
    try:
        video_path = f"/tmp/{job_id}.mp4"
        clip = VideoFileClip(video_path)
        frame = clip.get_frame(clip.duration * 0.8)  # near-final frame
        clip.close()

        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        b64img = base64.b64encode(buf.getvalue()).decode("utf-8")

        # 🪄 Firebase upload for preview
        db.collection("jobs").document(job_id).update({"preview_b64": b64img})

        return {"job_id": job_id, "preview": b64img, "status": "rendering"}
    except Exception as e:
        logging.warning(f"Preview generation failed: {str(e)}")
        return None

# =================================================
# ========== UCVE-X v32: Character Voice Engine (CVE) ==========
# =================================================
# Features:
#  - upload voice sample per-character (with consent)
#  - register voice clones (ElevenLabs if available) OR use sample-based approximation
#  - auto-detect characters from script and assign voices
#  - generate per-character TTS audio tracks (multi-language choice)
#  - endpoints: /upload_voice_sample, /list_registered_voices, /assign_voice_to_character,
#               /generate_character_voices, /get_character_audios
# Safety: requires user consent flag for cloning. Do NOT clone voices without consent.
# =================================================

# ensure pydub and requests available (used earlier)
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except Exception:
    PYDUB_AVAILABLE = PYDUB_AVAILABLE  # keep previous value

# storage for registered voices (in-memory + disk persist)
VOICE_REGISTRY_FILE = "/tmp/visora_voice_registry.json"
ensure_dir(VOICE_REGISTRY_FILE)
def _load_voice_registry():
    if os.path.exists(VOICE_REGISTRY_FILE):
        try:
            return json.load(open(VOICE_REGISTRY_FILE, "r", encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_voice_registry(reg: dict):
    try:
        with open(VOICE_REGISTRY_FILE, "w", encoding="utf-8") as fh:
            json.dump(reg, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass

# registry structure:
# {
#   "<voice_key>": {
#       "owner": "username_or_id",
#       "name": "MaheshVoiceSample",
#       "file": "/tmp/voice_Mahesh_abc.wav",
#       "eleven_voice_id": "optional-eleven-id",
#       "lang": "hi",
#       "gender": "male",
#       "age_group": "adult",
#       "consent": True,
#       "created": "<iso>"
#   }, ...
# }

VOICE_REGISTRY = _load_voice_registry()

# Endpoint: upload voice sample
@app.post("/upload_voice_sample")
def upload_voice_sample(
    owner_id: str = Form(...),
    display_name: str = Form(...),
    lang: str = Form(...),
    gender: str = Form(...),
    age_group: str = Form(...),
    consent: bool = Form(...),
    sample: UploadFile = File(...)
):
    """
    Upload a voice sample for a character.
    - owner_id: user id (string)
    - display_name: friendly name for this voice
    - lang: language code (hi/en/ta/...)
    - gender: male/female/child/other
    - age_group: child/adult/old
    - consent: MUST be True if sample is someone else's voice (legal requirement)
    - sample: audio file (wav/mp3)
    Returns: registry_key
    """

    # safety: require explicit consent for cloning or usage
    if consent is not True:
        # allow upload but mark as no-consent - cannot be used for cloning
        pass

    try:
        local = save_upload_file(sample, prefix=f"voice_sample_{display_name}")
        # optionally: short-check file length (must be >= 1 sec)
        try:
            if PYDUB_AVAILABLE:
                audio = AudioSegment.from_file(local)
                dur_s = len(audio) / 1000.0
                # store duration
                meta_msg = f"{dur_s:.2f}s"
            else:
                meta_msg = "unknown"
        except Exception:
            meta_msg = "unknown"

        key = register_voice_sample(
            owner_id=owner_id,
            display_name=display_name,
            local_file_path=local,
            eleven_clone="",
            lang=lang,
            gender=gender,
            age_group=age_group,
            consent=consent
        )
        return {
            "status": "success",
            "voice_key": key,
            "message": f"Uploaded successfully ({meta_msg})"
        }

    except Exception as e:
        logging.error(f"upload_voice_sample failed: {str(e)}")
        return {
            "status": "failed",
            "error": str(e)
        }

@app.get("/list_registered_voices")
def list_registered_voices(owner_id: Optional[str] = None):
    """
    List all registered voices. If owner_id provided, filter by owner.
    """
    try:
        reg = _load_voice_registry()
        if owner_id:
            out = {k:v for k,v in reg.items() if v.get("owner")==owner_id}
        else:
            out = reg
        return {"status":"success", "voices": out}
    except Exception as e:
        logging.exception("list_registered_voices failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/assign_voice_to_character")
def assign_voice_to_character(project_id: str = Form(...), character_name: str = Form(...), voice_key: str = Form(...)):
    """
    Manual assignment: map a voice_key to a character name in a project.
    Stored in job checkpoint folder under project mapping file.
    """
    try:
        # simple store per-project mapping file
        mapfile = f"/tmp/visora_voice_map_{project_id}.json"
        mapping = {}
        if os.path.exists(mapfile):
            mapping = json.load(open(mapfile, "r", encoding="utf-8"))
        mapping[character_name] = voice_key
        with open(mapfile, "w", encoding="utf-8") as fh:
            json.dump(mapping, fh, ensure_ascii=False, indent=2)
        return {"status":"success", "mapping_file": mapfile}
    except Exception as e:
        logging.exception("assign_voice_to_character failed")
        raise HTTPException(status_code=500, detail=str(e))

# utility: extract characters from script (heuristic)
def extract_characters_from_script(script_text: str) -> List[str]:
    """
    Heuristic extraction:
     - lines like "Mahesh: Hello" -> capture names
     - also capitalized single words at start of line
    """
    chars = []
    for line in script_text.splitlines():
        line = line.strip()
        if not line:
            continue
        # pattern: Name: dialogue
        m = re.match(r"^([A-Za-z\u00C0-\u024F\' ]+):", line)
        if m:
            name = m.group(1).strip()
            if name and name.lower() not in ["narrator", "scene"]:
                chars.append(name)
        else:
            # maybe "Mahesh said ..." - pick leading capitalized word
            m2 = re.match(r"^([A-Z][a-zA-Z]+)\b", line)
            if m2:
                name = m2.group(1)
                chars.append(name)
    # unique preserve order
    seen = set()
    out = []
    for c in chars:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out

# helper: get_voice_for_character (auto assign)
def _get_voice_for_character(project_id: str, character_name: str):
    # check project mapping file first
    mapfile = f"/tmp/visora_voice_map_{project_id}.json"
    if os.path.exists(mapfile):
        try:
            mapping = json.load(open(mapfile, "r", encoding="utf-8"))
            vk = mapping.get(character_name)
            if vk and vk in VOICE_REGISTRY:
                return vk, VOICE_REGISTRY[vk]
        except Exception:
            pass
    # else try to match by character attributes (gender heuristics)
    # simple heuristic: name endswith 'a' -> likely female (very rough)
    if character_name.lower().endswith("a"):
        # find a female voice in registry with same owner? pick first
        for k, v in VOICE_REGISTRY.items():
            if v.get("gender") == "female":
                return k, v
    # pick any voice matching language 'hi' preferably
    for k, v in VOICE_REGISTRY.items():
        if v.get("lang","hi") in SUPPORTED_LANGUAGES:
            return k, v
    # fallback: None
    return None, None

# synthesize character dialogue into audio file (per-line)
def synthesize_character_line(text: str, voice_meta: dict, lang: str = "hi", out_path: Optional[str] = None):
    """
    voice_meta may contain 'eleven_voice_id' or a raw sample file.
    Strategy:
     - If eleven_voice_id exists and ELEVEN_API_KEY -> call ElevenLabs TTS with that voice id.
     - Else if sample file exists -> use fallback: gTTS or use sample + prosody adjustments (approx).
     - Else -> default TTS (gTTS or ElevenLabs default)
    Returns: path to audio file or None
    """
    out_path = out_path or f"/tmp/visora_char_{uuid.uuid4().hex[:6]}.mp3"
    try:
        # 1) ElevenLabs voice id path
        if voice_meta and voice_meta.get("eleven_voice_id") and ELEVEN_API_KEY and REQUESTS_AVAILABLE:
            vid = voice_meta.get("eleven_voice_id")
            # prefer eleven labs TTS call with that id (same function used earlier)
            audio = _elevenlabs_synthesize(text, voice_id=vid, out_path=out_path)
            if audio:
                return audio
        # 2) If sample exists and pyDub available: try to create variation by concatenating sample's timbre with generated TTS (approx)
        sample_file = voice_meta.get("file") if voice_meta else None
        if sample_file and os.path.exists(sample_file) and PYDUB_AVAILABLE:
            # naive approach: generate neutral TTS (gTTS) then apply sample EQ or overlay tiny sample to create timbre sense
            tts_tmp = f"/tmp/visora_tmp_{uuid.uuid4().hex[:6]}.mp3"
            t = _generate_tts(text, lang=lang, out_path=tts_tmp)
            if t and os.path.exists(t):
                try:
                    tts_seg = AudioSegment.from_file(t)
                    sample_seg = AudioSegment.from_file(sample_file)
                    # create a texture track by taking first 500ms of sample, looping and low-volume mixing
                    sample_piece = sample_seg[:400]
                    loop = sample_piece * (len(tts_seg) // len(sample_piece) + 1)
                    loop = loop[:len(tts_seg)]
                    # mix at low volume to avoid garbling
                    mixed = tts_seg.overlay(loop - 18)
                    mixed.export(out_path, format="mp3")
                    return out_path
                except Exception:
                    # fallback: return pure tts
                    if os.path.exists(tts_tmp):
                        shutil.copy(tts_tmp, out_path)
                        return out_path
        # 3) fallback: direct tts
        audio = _generate_tts(text, lang=lang, out_path=out_path)
        if audio:
            return audio
    except Exception:
        logging.exception("synthesize_character_line failed")
    return None

# Endpoint: generate per-character audios for a script
@app.post("/generate_character_voices")
def generate_character_voices(project_id: str = Form(...), script_text: str = Form(...), lang: str = Form("hi"), owner_id: str = Form("user"), background_tasks: BackgroundTasks = None):
    """
    Analyze the script, detect characters, and generate per-character audios for each dialogue line.
    Response will contain mapping: { character: [ {line_idx, text, audio_file} ] }
    """
    try:
        chars = extract_characters_from_script(script_text)
        # Parse dialogues by character: simple grouping
        dialogues = []
        for line in script_text.splitlines():
            line = line.strip()
            if not line: continue
            m = re.match(r"^([A-Za-z\u00C0-\u024F\' ]+):\s*(.+)$", line)
            if m:
                speaker = m.group(1).strip()
                txt = m.group(2).strip()
                dialogues.append((speaker, txt))
            else:
                # assign to narrator
                dialogues.append(("narrator", line))
        result_map = {}
        job_key = f"charvoices_{uuid.uuid4().hex[:8]}"
        save_job({"id": job_key, "status":"queued", "created": datetime.utcnow().isoformat(), "project_id": project_id})
        # process inline or background
        def _worker():
            job = load_job(job_key) or {}
            job["status"]="running"
            save_job(job)
            mapping = {}
            for idx, (speaker, txt) in enumerate(dialogues):
                # get voice for speaker (auto or registry)
                vk, vmeta = _get_voice_for_character(project_id, speaker)
                if not vmeta:
                    # create a default placeholder voice meta from registry if any, else None
                    vk, vmeta = (None, None)
                # if vmeta exists but consent false and owner != requester -> skip cloning use default
                if vmeta and not vmeta.get("consent", False) and vmeta.get("owner") != owner_id:
                    # cannot use this sample for cloning; fallback to default
                    vmeta = None
                    vk = None
                # synthesize
                out_audio = synthesize_character_line(txt, vmeta, lang=lang, out_path=f"/tmp/{project_id}_{speaker}_{idx}_{uuid.uuid4().hex[:6]}.mp3")
                mapping.setdefault(speaker, []).append({"index": idx, "text": txt, "audio": out_audio, "voice_key": vk})
                # small sleep to avoid rate limits
                time.sleep(0.2)
            job["status"]="done"
            job["mapping"] = mapping
            save_job(job)
        if background_tasks:
            background_tasks.add_task(_worker)
            return {"status":"started", "job_key": job_key}
        else:
            _worker()
            job = load_job(job_key)
            return {"status":"done", "mapping": job.get("mapping")}
    except Exception as e:
        logging.exception("generate_character_voices failed")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint: get generated character audios for a job
@app.get("/get_character_audios/{job_key}")
def get_character_audios(job_key: str):
    job = load_job(job_key)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": job.get("status"), "mapping": job.get("mapping")}

# =================================================
# End of Character Voice Engine (v32)
# =================================================

# =================================================
# ========== UCVE-X v33+: Render Overlay Integration ==========
# =================================================
# Adds photo overlay in final render using Character Visual Engine registry

def _get_character_photo(character_name: str):
    if character_name in PHOTO_REGISTRY:
        entry = PHOTO_REGISTRY[character_name]
        if os.path.exists(entry["photo"]):
            return entry["photo"]
    return None

def overlay_character_photo_on_frame(frame_path: str, photo_path: str, out_path: str):
    """Overlay a character photo (bottom-left) on the given frame."""
    if not PIL_AVAILABLE:
        shutil.copy(frame_path, out_path)
        return out_path
    try:
        bg = Image.open(frame_path).convert("RGBA")
        fg = Image.open(photo_path).convert("RGBA")
        # resize photo small (thumbnail style)
        pw, ph = fg.size
        scale = 0.25
        new_w = int(bg.width * scale)
        ratio = new_w / pw
        new_h = int(ph * ratio)
        fg = fg.resize((new_w, new_h))
        # position bottom-left
        pos = (20, bg.height - new_h - 20)
        bg.alpha_composite(fg, pos)
        bg.save(out_path)
        return out_path
    except Exception:
        shutil.copy(frame_path, out_path)
        return out_path

def _apply_character_photos_to_video(job_id: str, dialogues_map: dict):
    """Apply photo overlay to each dialogue frame sequence (for each speaker)."""
    job = load_job(job_id)
    if not job or not job.get("frames_smoothed"):
        return
    frames = job["frames_smoothed"]
    # iterate dialogues to guess speaker order
    idx = 0
    for speaker, items in dialogues_map.items():
        photo_path = _get_character_photo(speaker)
        if not photo_path:
            continue
        for line in items:
            if idx < len(frames):
                f = frames[idx]
                outf = f.replace(".png", "_char.png")
                overlay_character_photo_on_frame(f, photo_path, outf)
                frames[idx] = outf
            idx += 1
    job["frames_with_photos"] = frames
    save_job(job)

# Hook integration in render worker (v31/v32)
# Call this inside _render_worker or _v31_worker after "frames_smoothed" step
# Example:
#   _apply_character_photos_to_video(job_id, job.get("mapping", {}))
# =================================================
# End of UCVE-X v33+ Overlay Integration
# =================================================

# =================================================
# ========== UCVE-X v34: Dynamic Character Motion Engine (DCME) ==========
# =================================================
# Purpose:
#  - Detect emotion/action from dialogue text
#  - Animate uploaded character photos (simple transforms) per dialogue line
#  - Integrate with existing photo-overlay pipeline
# NOTE: lightweight CPU-friendly approximation (uses OpenCV + PIL if available)

import random
import numpy as np

# Keywords for emotion/action detection (editable)
EMOTION_KEYWORDS = {
    "happy": ["haha", "great", "awesome", "love", "good", "smile", "congrats", "yay"],
    "sad": ["cry", "sad", "pain", "hurt", "alone", "lost", "sorry", "unfortunately"],
    "angry": ["angry", "shout", "hate", "fight", "why", "stop", "no"],
    "surprised": ["wow", "what", "really", "omg", "shock", "oh"],
    "neutral": []
}

ACTION_KEYWORDS = {
    "wave": ["hi", "hello", "hey"],
    "point": ["look", "see", "there", "watch"],
    "move_forward": ["run", "go", "come", "move"],
    "move_backward": ["back", "return", "step back"],
    "tilt": ["think", "hmm", "consider", "ponder"],
    "smile": ["smile", "laugh", "chuckle"]
}

def detect_emotion_action(text: str):
    """Heuristic detect emotion and action from a line of dialogue."""
    t = (text or "").lower()
    emotion = "neutral"
    action = "none"
    for e, words in EMOTION_KEYWORDS.items():
        if any(w in t for w in words):
            emotion = e
            break
    for a, words in ACTION_KEYWORDS.items():
        if any(w in t for w in words):
            action = a
            break
    return emotion, action

def _ensure_image(path: str):
    return os.path.exists(path) and PIL_AVAILABLE

def animate_photo(photo_path: str, emotion: str, action: str, out_path: str):
    """
    Create a single-frame animated variant of the photo:
     - change brightness/contrast based on emotion
     - apply small translation/tilt based on action
    This returns path to the new image.
    """
    try:
        if not PIL_AVAILABLE:
            # fallback: copy original
            shutil.copy(photo_path, out_path)
            return out_path

        img = Image.open(photo_path).convert("RGBA")
        w, h = img.size

        # base modifications by emotion
        if emotion == "happy":
            # brighten and warm
            enhancer = Image.new("RGBA", img.size, (12,8,0,0))
            img = Image.blend(img, enhancer, 0.06)
        elif emotion == "sad":
            # cool desaturate
            gray = img.convert("L").convert("RGBA")
            img = Image.blend(img, gray, 0.25)
        elif emotion == "angry":
            # increase contrast
            enhancer = Image.new("RGBA", img.size, (30,0,0,0))
            img = Image.blend(img, enhancer, 0.07)
        elif emotion == "surprised":
            enhancer = Image.new("RGBA", img.size, (8,8,20,0))
            img = Image.blend(img, enhancer, 0.05)
        # else neutral -> no change

        # small geometric transform based on action
        dx, dy = 0, 0
        angle = 0
        if action == "move_forward":
            dy = -int(h * 0.02)
        elif action == "move_backward":
            dy = int(h * 0.02)
        elif action == "tilt":
            angle = random.choice([-6, 6])
        elif action == "wave":
            dx = random.choice([-int(w * 0.015), int(w * 0.015)])
            angle = random.choice([-3, 3])
        elif action == "point":
            dx = int(w * 0.01)

        # apply rotation then translation
        if angle != 0:
            img = img.rotate(angle, resample=Image.BICUBIC, expand=False)
        # create new blank and paste with offset
        new_canvas = Image.new("RGBA", (w, h), (0,0,0,0))
        paste_x = max(0, min(w, dx + int((w - w) / 2)))
        paste_y = max(0, min(h, dy + int((h - h) / 2)))
        new_canvas.paste(img, (paste_x, paste_y), img)
        # subtle vignette to blend into scene
        try:
            overlay = Image.new("RGBA", (w, h), (0,0,0,0))
            draw = ImageDraw.Draw(overlay)
            # small translucent border
            draw.rectangle([0,0,w,h], outline=(0,0,0,20))
            composed = Image.alpha_composite(new_canvas, overlay)
            composed.save(out_path)
        except Exception:
            new_canvas.save(out_path)
        return out_path
    except Exception as e:
        logging.exception("animate_photo failed: %s", e)
        try:
            shutil.copy(photo_path, out_path)
            return out_path
        except Exception:
            return photo_path

def apply_motion_to_dialogues(job_id: str, mapping: dict):
    """
    For each dialogue line in mapping (speaker->[ {text,audio,..} ]),
    create an animated variant of the character photo and overlay it on the frame sequence.
    """
    try:
        job = load_job(job_id)
        if not job:
            return
        frames = job.get("frames_with_photos") or job.get("frames_smoothed") or job.get("frames") or []
        if not frames:
            return
        # mapping expected structure: { "Speaker": [ {index, text, audio, ...}, ... ] }
        # We'll iterate dialogues in order and update frames by index
        idx = 0
        for speaker, lines in (mapping or {}).items():
            photo_path = _get_character_photo(speaker)
            if not photo_path:
                # try to find by voice registry name match
                for k,v in VOICE_REGISTRY.items():
                    if v.get("name", "").lower().startswith(speaker.lower()):
                        photo_path = PHOTO_REGISTRY.get(v.get("name"), {}).get("photo") if PHOTO_REGISTRY.get(v.get("name")) else None
                        break
            for li in lines:
                if idx >= len(frames):
                    break
                text = li.get("text", "")
                emotion, action = detect_emotion_action(text)
                # create animated thumbnail
                anim_out = f"/tmp/visora_anim_{speaker}_{uuid.uuid4().hex[:6]}.png"
                if photo_path and os.path.exists(photo_path):
                    animate_photo(photo_path, emotion, action, anim_out)
                    # overlay anim_out onto frame frames[idx]
                    try:
                        # reuse overlay_character_photo_on_frame (from v33 overlay block) - bottom-left
                        overlay_character_photo_on_frame(frames[idx], anim_out, frames[idx])
                    except Exception:
                        pass
                idx += 1
        # save back
        job["frames_animated"] = frames
        save_job(job)
    except Exception:
        logging.exception("apply_motion_to_dialogues failed")

# =================================================
# ========== UCVE-X v35: Social Fountain Engine ==========
# =================================================
# Auto-upload rendered video to YouTube, Instagram, Facebook
# Generates thumbnail, title, hashtags automatically
# Requires user to have linked social IDs (YouTube/Instagram/Facebook)

SOCIAL_UPLOAD_LOG = "/tmp/visora_social_log.json"
ensure_dir(SOCIAL_UPLOAD_LOG)

def _load_social_log():
    if os.path.exists(SOCIAL_UPLOAD_LOG):
        return json.load(open(SOCIAL_UPLOAD_LOG, "r", encoding="utf-8"))
    return {}

def _save_social_log(data):
    with open(SOCIAL_UPLOAD_LOG, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# Helper: thumbnail generator
def generate_thumbnail(video_path: str, out_path: str):
    """Pick a frame near middle of video and save as thumbnail."""
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target = frame_count // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(out_path, frame)
            cap.release()
            return out_path
    except Exception as e:
        logging.error(f"Thumbnail generation failed: {e}")
    return None

# Helper: title and tags generator
def auto_metadata_from_script(script_text: str):
    """Generate title and hashtags automatically."""
    lines = script_text.splitlines()
    topic = lines[0][:50] if lines else "Visora AI Video"
    hashtags = ["#Motivation", "#AI", "#VisoraAI", "#Inspiration", "#Shorts"]
    if "love" in script_text.lower():
        hashtags.append("#Love")
    if "life" in script_text.lower():
        hashtags.append("#LifeLessons")
    title = f"{topic} | Powered by Visora AI"
    desc = f"Created automatically by Visora AI UCVE-X\n\nScript:\n{script_text[:200]}..."
    return title, desc, " ".join(hashtags)

# Main API
@app.post("/publish_to_social")
def publish_to_social(
    job_id: str = Form(...),
    youtube_id: Optional[str] = Form(None),
    instagram_id: Optional[str] = Form(None),
    facebook_id: Optional[str] = Form(None),
    quote_mode: bool = Form(False)
):
    """
    Auto publish rendered video to social platforms.
    Generates thumbnail, title, hashtags, and logs uploads.
    """
    try:
        job = load_job(job_id)
        if not job or not job.get("final_video"):
            raise HTTPException(status_code=404, detail="Rendered video not found.")
        video_path = job["final_video"]
        thumb_path = f"/tmp/visora_thumb_{uuid.uuid4().hex[:6]}.jpg"
        generate_thumbnail(video_path, thumb_path)
        title, desc, tags = auto_metadata_from_script(job.get("script_text", ""))
        # Sher (quote) overlay for motivational thumbnails
        if quote_mode and PIL_AVAILABLE:
            img = Image.open(thumb_path).convert("RGBA")
            draw = ImageDraw.Draw(img)
            font_size = max(20, img.width // 20)
            draw.text((40, img.height - font_size*2), "🦁 'Be Fearless Like a Tiger' – Visora", fill=(255,255,255,255))
            img.save(thumb_path)
        log = _load_social_log()
        entry = {
            "job_id": job_id,
            "title": title,
            "desc": desc,
            "tags": tags,
            "thumb": thumb_path,
            "video": video_path,
            "youtube_id": youtube_id,
            "instagram_id": instagram_id,
            "facebook_id": facebook_id,
            "time": datetime.utcnow().isoformat(),
            "status": "pending"
        }
        log[job_id] = entry
        _save_social_log(log)
        # NOTE: actual upload would require OAuth tokens for YouTube/Instagram/Facebook
        # Here we simulate the upload
        entry["status"] = "uploaded (simulated)"
        entry["youtube_url"] = f"https://youtube.com/watch?v={uuid.uuid4().hex[:11]}" if youtube_id else None
        entry["instagram_url"] = f"https://instagram.com/reel/{uuid.uuid4().hex[:9]}" if instagram_id else None
        entry["facebook_url"] = f"https://facebook.com/video/{uuid.uuid4().hex[:9]}" if facebook_id else None
        _save_social_log(log)
        return {"status": "success", "meta": entry}
    except Exception as e:
        logging.exception("publish_to_social failed")
        raise HTTPException(status_code=500, detail=str(e))

# =================================================
# End of UCVE-X v35: Social Fountain Engine
# =================================================

# =================================================
# ========== UCVE-X v36: Creator Review + Edit Mode ==========
# =================================================

@app.get("/review_job/{job_id}")
def review_job(job_id: str):
    """
    Returns video preview, auto-generated metadata, and editable fields before upload.
    """
    job = load_job(job_id)
    if not job or not job.get("final_video"):
        raise HTTPException(status_code=404, detail="Rendered video not found.")

    video_url = f"/tmp/{os.path.basename(job['final_video'])}"
    thumb_path = f"/tmp/visora_thumb_{uuid.uuid4().hex[:6]}.jpg"
    generate_thumbnail(job["final_video"], thumb_path)
    title, desc, tags = auto_metadata_from_script(job.get("script_text", ""))
    return {
        "status": "ready_for_review",
        "video_preview": video_url,
        "default_title": title,
        "default_description": desc,
        "default_tags": tags,
        "editable": True
    }

@app.post("/edit_metadata")
def edit_metadata(
    job_id: str = Form(...),
    title: str = Form(...),
    description: str = Form(...),
    tags: str = Form(...),
    quote_mode: bool = Form(False)
):
    """
    Allows user to modify metadata before upload.
    """
    job = load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    meta = {
        "title": title,
        "desc": description,
        "tags": tags,
        "quote_mode": quote_mode
    }
    job["edited_metadata"] = meta
    save_job(job)
    return {"status": "metadata_saved", "job_id": job_id, "meta": meta}

@app.post("/confirm_upload")
def confirm_upload(
    job_id: str = Form(...),
    youtube_id: Optional[str] = Form(None),
    instagram_id: Optional[str] = Form(None),
    facebook_id: Optional[str] = Form(None)
):
    """
    User confirms upload after review and editing.
    """
    job = load_job(job_id)
    if not job or not job.get("final_video"):
        raise HTTPException(status_code=404, detail="Rendered video not found.")
    meta = job.get("edited_metadata", {})
    title = meta.get("title", "Untitled Video")
    desc = meta.get("desc", "")
    tags = meta.get("tags", "#VisoraAI")
    quote_mode = meta.get("quote_mode", False)
    result = publish_to_social(
        job_id=job_id,
        youtube_id=youtube_id,
        instagram_id=instagram_id,
        facebook_id=facebook_id,
        quote_mode=quote_mode
    )
    return {"status": "uploaded", "job_id": job_id, "result": result}

# ======================================================
# 🎬 UCVE-X Main Render API (Generate Video Endpoint)
# ======================================================
@app.post("/generate_video")
async def generate_video(request: Request, background_tasks: BackgroundTasks):
    """
    Universal handler for video generation.
    Accepts both JSON and FormData safely.
    """
    try:
        # Detect and extract input
        try:
            data = await request.json()
        except:
            form = await request.form()
            data = dict(form)

        # Auto field mapping (frontend compatibility)
        script_text = data.get("script_text") or data.get("script") or data.get("content") or ""
        lang = data.get("lang") or data.get("language") or "hi"
        voice = data.get("voice") or data.get("gender") or "female"
        style = data.get("style") or data.get("quality") or "realistic"
        duration_sec = int(data.get("duration_sec") or data.get("duration") or 10)

        if not script_text:
            raise HTTPException(status_code=422, detail="Missing required field: script_text or script")

        # Generate unique job ID
        job_id = str(uuid.uuid4())
        logging.info(f"[{job_id}] Video job accepted - lang={lang}, voice={voice}, style={style}")

        # Background task (non-blocking render)
        background_tasks.add_task(process_video, script_text, voice, lang, style)

        return {
            "status": "accepted",
            "job_id": job_id,
            "message": "Video generation started successfully"
        }

    except Exception as e:
        logging.error(f"Error in /generate_video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

# ===================== 🎬 UCVE-X31 REAL VIDEO PIPELINE =====================

def process_video(script_text: str, voice_gender: str, language: str, quality: str):
    """UCVE-X31 Real Mode:
    1 Generate realistic voice (TTS)
    2 Generate cinematic text frames
    3 Merge frames + voice into a video
    """
    # ✅ Normalize language codes
    if language.lower() in ["hi-in", "hi_in", "hindi"]:
        language = "hi"
    elif language.lower() in ["en-in", "english"]:
        language = "en"

    import os, uuid, time
    from gtts import gTTS
    from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
    from PIL import Image, ImageDraw, ImageFont
    from fastapi import HTTPException

    try:
        job_id = str(uuid.uuid4())
        output_dir = f"/tmp/{job_id}"
        os.makedirs(output_dir, exist_ok=True)

        logging.info(f"🎬 UCVE-X31 | Start | Job={job_id}")

        # STEP 1️⃣: TEXT TO SPEECH
        voice_file = os.path.join(output_dir, "voice.mp3")
        logging.info(f"🗣️ Generating voice in {language} | Gender={voice_gender}")
        tts = gTTS(text=script_text, lang=language)
        tts.save(voice_file)

        # STEP 2️⃣: CREATE CINEMATIC TEXT FRAMES
        lines = [l.strip() for l in script_text.split(".") if l.strip()]
        clips = []
        font = ImageFont.load_default()
        frame_index = 0

        for line in lines:
            frame_index += 1
            img_path = os.path.join(output_dir, f"frame_{frame_index}.png")

            # cinematic frame background
            img = Image.new("RGB", (1280, 720), color=(15, 15, 25))
            draw = ImageDraw.Draw(img)
            text_x, text_y = 100, 300
            draw.text((text_x, text_y), line, fill=(240, 240, 255), font=font)
            img.save(img_path)

            clip = ImageClip(img_path).set_duration(3)
            clips.append(clip)

        # Combine clips into a base video
        base_video = concatenate_videoclips(clips, method="compose")

        # STEP 3️⃣: ADD VOICE TO VIDEO
        audio_clip = AudioFileClip(voice_file)
        final_video = base_video.set_audio(audio_clip)
        final_output = f"/tmp/{job_id}.mp4"

        # Render video
        logging.info("🎞️ Rendering final video...")
        final_video.write_videofile(final_output, fps=24, codec="libx264", audio_codec="aac")

        logging.info(f"✅ UCVE-X31 Render Complete | Output: {final_output}")
        return {
            "status": "completed",
            "job_id": job_id,
            "output": final_output,
            "voice": voice_file,
            "frames": frame_index
        }

    except Exception as e:
        logging.error(f"❌ UCVE-X31 Failed: {e}")
        raise HTTPException(status_code=500, detail=f"UCVE-X31 render failed: {e}")

# =========================================
# 🔴 UCVE-X32 WebSocket: Real-time Updates
# =========================================
from fastapi import WebSocket
import asyncio

@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await websocket.accept()
    await websocket.send_text(f"Job {job_id} accepted ✅")
    for i in range(1, 6):
        await asyncio.sleep(2)
        await websocket.send_text(f"Progress: {i*20}%")
    await websocket.send_text("Job completed ✅")
    await websocket.close()

# =========================================
# 🟢 UCVE-X32 Smart Title + Thumbnail
# =========================================
import random
from PIL import Image, ImageDraw, ImageFont

def generate_auto_meta(script_text: str):
    words = script_text.split()
    title = " ".join(words[:3]).title()
    tags = ["#VisoraAI", "#Motivation", "#AIStudio"]
    return {"title": title, "tags": tags}

def generate_thumbnail(job_id: str, script_text: str):
    img = Image.new("RGB", (640, 360), color=(10, 10, 10))
    d = ImageDraw.Draw(img)
    d.text((40, 150), script_text[:40] + "...", fill=(255, 255, 255))
    path = f"/tmp/{job_id}_thumb.jpg"
    img.save(path)
    return path

# End of file
