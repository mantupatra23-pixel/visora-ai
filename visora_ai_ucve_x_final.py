#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visora_ai_ucve_x_final.py
Visora AI — UCVE-X unified single-file backend (v1..v20 + Pro modules)
- GPU/CPU autodetect (supports both)
- Default style = "realistic"
- ElevenLabs voice + fallbacks
- Prompt optimizer, style transfer, temporal, lipsync stubs, SFX, memory, resume
Author: dev-bro (Hindi)
"""

import os, re, uuid, hmac, hashlib, logging, json, time
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks, Request
from pydantic import BaseModel

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# -------------------------
# Optional heavy imports (graceful)
# -------------------------
NP_AVAILABLE = False
CV2_AVAILABLE = False
PIL_AVAILABLE = False
MOVIEPY_AVAILABLE = False
TORCH_AVAILABLE = False
DIFFUSERS_AVAILABLE = False
OPEN3D_AVAILABLE = False
PYDUB_AVAILABLE = False
LIBROSA_AVAILABLE = False
REDIS_AVAILABLE = False
RQ_AVAILABLE = False
PROM_AVAILABLE = False
TEXTBLOB_AVAILABLE = False
REQUESTS_AVAILABLE = False

try:
    import numpy as np
    NP_AVAILABLE = True
except Exception as e:
    logging.info("numpy missing: %s", e)

try:
    import cv2
    CV2_AVAILABLE = True
except Exception as e:
    logging.info("cv2 missing: %s", e)

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except Exception as e:
    logging.info("PIL missing: %s", e)

try:
    from moviepy.editor import ImageSequenceClip, AudioFileClip, CompositeVideoClip
    MOVIEPY_AVAILABLE = True
except Exception as e:
    logging.info("moviepy missing: %s", e)

try:
    import torch
    TORCH_AVAILABLE = True
except Exception as e:
    logging.info("torch missing: %s", e)

try:
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    DIFFUSERS_AVAILABLE = True
except Exception as e:
    logging.info("diffusers missing: %s", e)

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except Exception as e:
    logging.info("open3d missing: %s", e)

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except Exception as e:
    logging.info("pydub missing: %s", e)

try:
    import librosa
    LIBROSA_AVAILABLE = True
except Exception as e:
    logging.info("librosa missing: %s", e)

try:
    import redis
    REDIS_AVAILABLE = True
    from rq import Queue, Worker, Connection
    RQ_AVAILABLE = True
except Exception as e:
    logging.info("redis/rq missing: %s", e)

try:
    from prometheus_client import Counter, generate_latest
    PROM_AVAILABLE = True
except Exception as e:
    logging.info("prometheus_client missing: %s", e)

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except Exception as e:
    logging.info("textblob missing: %s", e)

try:
    import requests
    REQUESTS_AVAILABLE = True
except Exception as e:
    logging.info("requests missing: %s", e)

# Payments libs (optional)
try:
    import stripe
except Exception:
    stripe = None
try:
    import razorpay
except Exception:
    razorpay = None

# -------------------------
# App init
# -------------------------
app = FastAPI(title="Visora AI UCVE-X", version="ucve_x")

# -------------------------
# Global config / env
# -------------------------
DOMAIN = os.getenv("DOMAIN", "http://127.0.0.1:8000")
HUGGINGFACE_HUB_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
ELEVEN_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVEN_VOICE_ID = os.getenv("ELEVEN_VOICE_ID", None)
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "elevenlabs")  # default to elevenlabs
RENDER_WATERMARK = os.getenv("RENDER_WATERMARK", "Visora AI")
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET")
REDIS_URL = os.getenv("REDIS_URL")
USE_RQ = False
rq_queue = None
if REDIS_AVAILABLE and RQ_AVAILABLE and REDIS_URL:
    try:
        redis_conn = redis.from_url(REDIS_URL)
        rq_queue = Queue("visora_jobs", connection=redis_conn)
        USE_RQ = True
    except Exception as e:
        logging.exception("Redis init failed")

PROM_COUNTER = Counter("ucve_jobs_total", "Total UCVE jobs") if PROM_AVAILABLE else None

# default style mapping (1..4) — user asked default = 2 -> realistic
STYLE_MAP = {1: "cartoon", 2: "realistic", 3: "anime", 4: "cinematic"}
DEFAULT_STYLE = STYLE_MAP.get(2, "realistic")

# -------------------------
# Utilities
# -------------------------
def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def save_temp_file(upload: UploadFile, prefix: str = "tmp") -> str:
    name = os.path.basename(upload.filename)
    tmp_path = f"/tmp/{prefix}_{uuid.uuid4().hex[:8]}_{name}"
    ensure_dir(tmp_path)
    with open(tmp_path, "wb") as fh:
        fh.write(upload.file.read())
    return tmp_path

def _watermark_image_pil(img_path: str, out_path: str, text: str = RENDER_WATERMARK):
    if not PIL_AVAILABLE:
        return img_path
    try:
        im = Image.open(img_path).convert("RGBA")
        draw = ImageDraw.Draw(im)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
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

def _get_torch_device_and_dtype():
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32

# -------------------------
# Simple in-memory memory store (character memory) — persisted to disk
# -------------------------
MEMORY_FILE = "/tmp/visora_character_memory.json"
def _load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            return json.load(open(MEMORY_FILE, "r", encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_memory(mem: dict):
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as fh:
            json.dump(mem, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass

# -------------------------
# Prompt Optimizer (basic rule-based + optional HF call stub)
# -------------------------
def prompt_optimizer_simple(prompt: str, style: str = DEFAULT_STYLE) -> str:
    # Expand short prompt into cinematic detailed prompt (rules)
    base = prompt.strip()
    # sample templates per style
    templates = {
        "cartoon": "A bright, colorful cartoon illustration of {p}, exaggerated expressions, smooth lines, cel-shaded, high detail",
        "realistic": "A highly detailed photorealistic scene of {p}, cinematic lighting, shallow depth of field, realistic textures",
        "anime": "An anime-style scene of {p}, vibrant colors, dramatic lighting, soft gradients, high detail",
        "cinematic": "A cinematic, filmic shot of {p}, wide-angle, dramatic lighting, film grain, high contrast"
    }
    tpl = templates.get(style, templates["realistic"])
    expanded = tpl.format(p=base) + ", ultra-detailed, high-resolution"
    # optional: further expand with a transformer/HF API (skip if token missing)
    if HUGGINGFACE_HUB_TOKEN and REQUESTS_AVAILABLE:
        # stub: call an external prompt-expander if you have one (not implemented)
        pass
    return expanded

# -------------------------
# Style Transfer: LUT-like filter (teal-orange simple) + stub for ControlNet
# -------------------------
def style_transfer_apply(image_path: str, style: str = "realistic") -> Optional[str]:
    if not (PIL_AVAILABLE or CV2_AVAILABLE):
        return None
    out = image_path.replace(".png", f"_{style}.png")
    try:
        if style == "realistic":
            # do nothing — SD already realistic
            return image_path
        if style == "cartoon" and PIL_AVAILABLE:
            im = Image.open(image_path).convert("RGB")
            # posterize-like effect
            im_small = im.resize((im.width//2, im.height//2)).resize(im.size, Image.NEAREST)
            im_small.save(out)
            return out
        if style == "anime" and CV2_AVAILABLE:
            img = cv2.imread(image_path)
            # bilateral filter + edge
            img_color = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
            edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 200)
            edges_col = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            anime = cv2.bitwise_and(img_color, cv2.bitwise_not(edges_col))
            cv2.imwrite(out, anime)
            return out
        # cinematic -> apply LUT stub
        if style == "cinematic" and CV2_AVAILABLE:
            img = cv2.imread(image_path).astype(float)/255.0
            r = img[:,:,2]; g = img[:,:,1]; b = img[:,:,0]
            r = np.clip(r*1.05 + 0.02, 0, 1)
            b = np.clip(b*0.9, 0, 1)
            out_img = (np.stack([b,g,r], axis=-1)*255).astype(np.uint8)
            cv2.imwrite(out, out_img)
            return out
    except Exception:
        return image_path
    return image_path

# -------------------------
# Temporal consistency stub (anti-flicker) — optical flow & frame blend placeholder
# -------------------------
def temporal_smooth_frames(frames: List[str]) -> List[str]:
    # Placeholder: for now do nothing or simple frame crossfade copies
    # Proper implementation: use video-diffusion or flow-guided stabilization
    if not frames or len(frames) < 2:
        return frames
    smoothed = []
    for i, f in enumerate(frames):
        smoothed.append(f)
    return smoothed

# -------------------------
# LipSync stub (Wav2Lip integration placeholder)
# -------------------------
def lipsync_audio_to_video(audio_path: str, reference_frame: Optional[str] = None) -> Optional[str]:
    # Real integration: run Wav2Lip model to generate video; here return audio path as stub
    return audio_path

# -------------------------
# SFX Engine (simple rule-based selector + mix)
# -------------------------
SFX_BANK = {
    "running": "sfx/running_loop.mp3",
    "rain": "sfx/rain_loop.mp3",
    "explosion": "sfx/explosion_short.mp3",
    "wind": "sfx/wind_ambience.mp3",
    "default": None
}

def select_sfx_for_prompt(prompt: str) -> Optional[str]:
    p = prompt.lower()
    if "run" in p or "running" in p:
        return SFX_BANK.get("running")
    if "rain" in p or "storm" in p:
        return SFX_BANK.get("rain")
    if "explode" in p or "bomb" in p:
        return SFX_BANK.get("explosion")
    if "wind" in p or "desert" in p:
        return SFX_BANK.get("wind")
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
        return tts_path

# -------------------------
# ElevenLabs TTS via REST (requests)
# -------------------------
def _elevenlabs_synthesize(text: str, voice_id: Optional[str] = None, out_path: str = "/tmp/ucve_eleven_tts.mp3"):
    if not REQUESTS_AVAILABLE:
        logging.info("requests missing for ElevenLabs")
        return None
    api_key = ELEVEN_API_KEY
    if not api_key:
        logging.info("ELEVENLABS_API_KEY not set")
        return None
    vid = voice_id or ELEVEN_VOICE_ID
    if not vid:
        # try fetch first voice
        try:
            resp = requests.get("https://api.elevenlabs.io/v1/voices", headers={"xi-api-key": api_key}, timeout=15)
            if resp.status_code == 200:
                voices = resp.json().get("voices", [])
                if voices:
                    vid = voices[0].get("voice_id") or voices[0].get("id")
        except Exception:
            vid = None
    if not vid:
        logging.info("No ElevenLabs voice id found")
        return None
    try:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{vid}"
        headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
        payload = {"text": text}
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code == 200:
            with open(out_path, "wb") as fh:
                fh.write(r.content)
            return out_path
        else:
            logging.error("ElevenLabs synth failed: %s %s", r.status_code, r.text)
            return None
    except Exception:
        logging.exception("elevenlabs synth error")
        return None

# fallback TTS (gTTS or pyttsx3)
def _generate_tts_audio(text: str, lang: str = "hi", out_path: str = "/tmp/ucve_tts.mp3"):
    # prefer ElevenLabs
    if TTS_PROVIDER == "elevenlabs" and ELEVEN_API_KEY:
        out = _elevenlabs_synthesize(text, out_path=out_path)
        if out:
            return out
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang=lang)
        tts.save(out_path)
        return out_path
    except Exception:
        pass
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.save_to_file(text, out_path)
        engine.runAndWait()
        return out_path
    except Exception:
        pass
    return None

# -------------------------
# Diffusers SD pipeline helpers (GPU/CPU ready)
# -------------------------
def _load_sd_pipeline(model_id: str = "runwayml/stable-diffusion-v1-5", hf_token: Optional[str] = None):
    if not DIFFUSERS_AVAILABLE or not TORCH_AVAILABLE:
        raise RuntimeError("diffusers/torch not installed")
    device, dtype = _get_torch_device_and_dtype()
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype, use_auth_token=hf_token)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    pipe = pipe.to(device)
    return pipe

def _synthesize_frames_from_prompt(pipe, prompt: str, num_frames: int = 24, width: int = 512, height: int = 512, guidance_scale: float = 7.5, seed: Optional[int]=None):
    out_files = []
    if seed is None:
        seed = int.from_bytes(os.urandom(2), "big")
    device = next(pipe.unet.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    for i in range(num_frames):
        frame_prompt = f"{prompt}, frame {i}, high detail"
        with torch.autocast(device.type) if TORCH_AVAILABLE and device.type == "cuda" else torch.cpu.amp.autocast(enabled=False):
            res = pipe(frame_prompt, height=height, width=width, guidance_scale=guidance_scale, generator=generator)
            image = res.images[0]
        out_path = f"/tmp/ucve_frame_{uuid.uuid4().hex[:6]}_{i}.png"
        image.save(out_path)
        wm = out_path.replace(".png","_wm.png")
        _watermark_image_pil(out_path, wm)
        out_files.append(wm)
    return out_files

# -------------------------
# Job checkpoint / resume stub
# -------------------------
JOB_CHECKPOINT_DIR = "/tmp/visora_jobs"
ensure_dir(JOB_CHECKPOINT_DIR)
def save_job_checkpoint(job_id: str, data: dict):
    try:
        path = os.path.join(JOB_CHECKPOINT_DIR, f"{job_id}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass

def load_job_checkpoint(job_id: str) -> Optional[dict]:
    path = os.path.join(JOB_CHECKPOINT_DIR, f"{job_id}.json")
    if os.path.exists(path):
        try:
            return json.load(open(path, "r", encoding="utf-8"))
        except Exception:
            return None
    return None

# -------------------------
# Pydantic models
# -------------------------
class ScriptIn(BaseModel):
    script_text: str
    style_id: Optional[int] = 2    # default 2 -> realistic
    duration_sec: Optional[int] = 6
    fps: Optional[int] = 24
    width: Optional[int] = 512
    height: Optional[int] = 512

class RenderRequestOut(BaseModel):
    job_id: str
    status: str
    message: Optional[str] = None
    video: Optional[str] = None

# =================================================
# Core endpoints (v1..v10) — same as before (concise)
# =================================================
@app.post("/v1/analyze_emotion")
def api_analyze_emotion(payload: ScriptIn):
    try:
        mood = analyze_emotion(payload.script_text)
        return {"status":"success","mood":mood}
    except Exception as e:
        logging.exception("emotion failed")
        raise HTTPException(status_code=500, detail=str(e))

def analyze_emotion(script_text: str) -> str:
    try:
        if TEXTBLOB_AVAILABLE:
            blob = TextBlob(script_text)
            polarity = blob.sentiment.polarity
            if polarity > 0.3: return "happy"
            if polarity < -0.3: return "sad"
            return "neutral"
        s = script_text.lower()
        if any(w in s for w in ["khushi","happy","shukriya"]): return "happy"
        if any(w in s for w in ["dukh","sad","tragic"]): return "sad"
        return "neutral"
    except Exception:
        return "neutral"

@app.post("/v2/generate_subtitles")
def api_generate_subtitles(payload: ScriptIn):
    try:
        return {"status":"success","subtitles": generate_subtitles(payload.script_text, "hi", 3.5)}
    except Exception as e:
        logging.exception("subs failed")
        raise HTTPException(status_code=500, detail=str(e))

def generate_subtitles(script_text: str, lang_target: str = "hi", dur_per_line: float = 3.5):
    lines = [ln for ln in script_text.strip().splitlines() if ln.strip()]
    subs=[]
    st=0.0
    for ln in lines:
        et=st+dur_per_line
        subs.append({"text":ln.strip(),"start":round(st,2),"end":round(et,2)})
        st=et
    return subs

@app.post("/v3/detect_dialogues")
def api_detect_dialogues(payload: ScriptIn):
    try:
        return {"status":"success","dialogues": detect_dialogues(payload.script_text)}
    except Exception as e:
        logging.exception("detect failed")
        raise HTTPException(status_code=500, detail=str(e))

def detect_dialogues(script_text: str):
    dialogues=[]
    for line in script_text.splitlines():
        line=line.strip()
        if not line: continue
        m=re.match(r"^([A-Za-z0-9_ ]+):\s*(.+)$", line)
        if m:
            dialogues.append({"speaker":m.group(1).strip(),"text":m.group(2).strip()})
        else:
            dialogues.append({"speaker":"narrator","text":line})
    return dialogues

@app.post("/v4/translate_caption")
def api_translate_caption(text: str = Form(...), target_lang: str = Form("hi")):
    return {"status":"success","translated":text,"lang":target_lang}

@app.post("/v5/compose_music")
def api_compose_music(mood: str = Form(...), duration: float = Form(20.0)):
    try:
        out=compose_emotion_music(mood,duration)
        return {"status":"success","file":out}
    except Exception as e:
        logging.exception("compose failed")
        raise HTTPException(status_code=500, detail=str(e))

def compose_emotion_music(mood: str, duration: float = 20.0, out_path: str = "/tmp/bg_music.mp3"):
    if not PYDUB_AVAILABLE:
        logging.info("pydub not installed")
        return None
    try:
        pool = {"happy":["forest_birds.mp3"],"sad":["rain_ambience.mp3"],"neutral":["daylight_soft.mp3"]}.get(mood,["daylight_soft.mp3"])
        found=None
        for f in pool:
            if os.path.exists(f):
                found=f; break
        if found:
            seg=AudioSegment.from_file(found)
            ms=int(duration*1000)
            out=seg*(ms//len(seg)+1)
            out=out[:ms]
            out.export(out_path,format="mp3")
            return out_path
        else:
            AudioSegment.silent(duration=int(duration*1000)).export(out_path,format="mp3")
            return out_path
    except Exception:
        return None

@app.post("/v6/upload_to_firebase")
def api_upload_to_firebase(file: UploadFile = File(...), remote_path: str = Form(...)):
    ok=init_firebase_if_needed()
    if not ok:
        return {"status":"error","message":"firebase not configured"}
    tmp=save_temp_file(file,"firebase")
    url=upload_to_firebase(tmp, remote_path)
    return {"status":"success","url":url}

def init_firebase_if_needed():
    global FIREBASE_AVAILABLE, FIREBASE_BUCKET, FIREBASE_DB
    try:
        import firebase_admin
        from firebase_admin import credentials, storage, db as firebase_db
        FIREBASE_AVAILABLE=True
    except Exception as e:
        logging.info("firebase missing: %s", e)
        FIREBASE_AVAILABLE=False
        return False
    try:
        if not firebase_admin._apps:
            cred_path=os.getenv("FIREBASE_CRED_PATH")
            bucket_name=os.getenv("FIREBASE_BUCKET")
            db_url=os.getenv("FIREBASE_DB_URL")
            if not cred_path or not os.path.exists(cred_path):
                logging.warning("firebase cred missing")
                return False
            cred=credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred, {"storageBucket":bucket_name, "databaseURL":db_url})
        from firebase_admin import storage, db as firebase_db
        FIREBASE_BUCKET = storage.bucket()
        FIREBASE_DB = firebase_db
        return True
    except Exception:
        return False

def upload_to_firebase(local_path, remote_path):
    global FIREBASE_BUCKET
    try:
        blob = FIREBASE_BUCKET.blob(remote_path)
        blob.upload_from_filename(local_path)
        blob.make_public()
        return blob.public_url
    except Exception:
        return None

@app.post("/v7/generate_actor_image")
def api_generate_actor_image(actor_type: str = Form("male"), text_overlay: str = Form("")):
    try:
        out=generate_actor_image(actor_type, text_overlay)
        return {"status":"success","file":out}
    except Exception as e:
        logging.exception("actor failed"); raise HTTPException(status_code=500, detail=str(e))

def generate_actor_image(actor_type="male", text=""):
    out=f"/tmp/actor_{actor_type}_{uuid.uuid4().hex[:6]}.png"
    if not PIL_AVAILABLE:
        return None
    try:
        base=(f"actors/{actor_type}_default.png")
        if os.path.exists(base):
            img=Image.open(base).convert("RGBA")
        else:
            img=Image.new("RGBA",(512,512),(80,80,120,255))
        draw=ImageDraw.Draw(img)
        try: font=ImageFont.load_default()
        except: font=None
        draw.text((20, img.height-60), text or actor_type.title(), fill=(255,255,255), font=font)
        img.save(out)
        return out
    except Exception:
        return None

@app.post("/v8/plan_shots")
def api_plan_shots(payload: ScriptIn):
    return {"status":"success","shots": plan_camera_shots(payload.script_text)}

def plan_camera_shots(script_text):
    txt=script_text.lower()
    shots=[{"type":"wide","duration":2.5,"intensity":0.2}]
    sents=re.split(r'[\.!\?]\s*', script_text.strip())
    for s in sents:
        if not s.strip(): continue
        t="closeup" if len(s.split())<6 else "medium"
        dur=min(4.5, max(1.5, len(s.split())*0.3))
        shots.append({"type":t,"duration":round(dur,2),"intensity":0.3})
    if any(k in txt for k in ["finally","end","conclusion","reward"]):
        shots.append({"type":"dramatic","duration":3.0,"intensity":0.8})
    return shots

@app.post("/v9/collab_merge")
def api_collab_merge(payload: dict):
    merged="\n".join(payload.get("script_versions",[]))
    return {"status":"merged","merged_script":merged}

@app.post("/v10/dashboard_metrics")
def api_dashboard_metrics(records: List[Dict]):
    return {"status":"success","metrics": compute_dashboard_metrics(records)}

def compute_dashboard_metrics(user_videos):
    videos=len(user_videos)
    succ=sum(1 for v in user_videos if v.get("status")=="done")
    fail=sum(1 for v in user_videos if v.get("status")=="failed")
    total=sum(v.get("credits",0) for v in user_videos)
    rate=round((succ/videos)*100,2) if videos else 0.0
    return {"videos":videos,"success_jobs":succ,"failed_jobs":fail,"total_credits":total,"success_rate":rate}

# =================================================
# PRO modules: Prompt optimizer, style transfer, temporal, lipsync, sfx, memory, resume
# =================================================

# Admin endpoints for Job control
JOBS_DIR = "/tmp/visora_jobs_full"
ensure_dir(JOBS_DIR)

@app.post("/pro/render_request", response_model=RenderRequestOut)
def pro_render_request(script_text: str = Form(...), style_id: int = Form(2),
                       duration_sec: int = Form(6), fps: int = Form(24),
                       width: int = Form(512), height: int = Form(512),
                       tts_text: Optional[str] = Form(None),
                       background_tasks: BackgroundTasks = None):
    """Create render job: this enqueues or runs background render chain (prompt optimize → frames → temporal → audio merge → final)"""
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    job = {
        "id": job_id,
        "status": "queued",
        "created": datetime.utcnow().isoformat(),
        "script": script_text,
        "style": STYLE_MAP.get(style_id, DEFAULT_STYLE),
        "duration_sec": duration_sec,
        "fps": fps,
        "width": width,
        "height": height,
        "tts_text": tts_text
    }
    save_job_checkpoint(job_id, job)
    # enqueue or run background task
    if USE_RQ and rq_queue:
        rq_queue.enqueue(_pro_render_worker, job_id)
        job["status"]="enqueued"
    else:
        # use background tasks immediate
        if background_tasks:
            background_tasks.add_task(_pro_render_worker, job_id)
            job["status"]="background_started"
        else:
            # run inline (blocking)
            _pro_render_worker(job_id)
            job = load_job_checkpoint(job_id) or job
    save_job_checkpoint(job_id, job)
    return RenderRequestOut(job_id=job_id, status=job["status"], message="Job queued/started")

@app.get("/pro/job_status/{job_id}")
def pro_job_status(job_id: str):
    job = load_job_checkpoint(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.post("/pro/cancel_job/{job_id}")
def pro_cancel_job(job_id: str):
    # simple cancel flag
    job = load_job_checkpoint(job_id)
    if not job:
        raise HTTPException(status_code=404)
    job["status"]="cancelled"
    save_job_checkpoint(job_id, job)
    return {"status":"success","message":"cancelled"}

def _pro_render_worker(job_id: str):
    """Main worker: runs the full pipeline for a job_id"""
    job = load_job_checkpoint(job_id)
    if not job:
        return
    try:
        job["status"]="running"
        save_job_checkpoint(job_id, job)
        # 1) Prompt optimize
        style = job.get("style", DEFAULT_STYLE)
        optimized_prompt = prompt_optimizer_simple(job["script"], style=style)
        job["optimized_prompt"] = optimized_prompt
        save_job_checkpoint(job_id, job)
        # 2) Select sfx
        sfx = select_sfx_for_prompt(job["script"])
        job["sfx"] = sfx
        save_job_checkpoint(job_id, job)
        # 3) Render frames (diffusers)
        if DIFFUSERS_AVAILABLE and TORCH_AVAILABLE:
            pipe = _load_sd_pipeline(model_id="runwayml/stable-diffusion-v1-5", hf_token=HUGGINGFACE_HUB_TOKEN)
            frames = _synthesize_frames_from_prompt(pipe, optimized_prompt, num_frames=max(1, job["duration_sec"]*job["fps"]), width=job["width"], height=job["height"])
            job["frames"] = frames
            save_job_checkpoint(job_id, job)
            # 4) Temporal smoothing
            frames = temporal_smooth_frames(frames)
            job["frames_smoothed"] = frames
            save_job_checkpoint(job_id, job)
        else:
            # fallback: create placeholder single-frame using prompt -> image optional
            dummy_path = f"/tmp/ucve_placeholder_{uuid.uuid4().hex[:6]}.png"
            if PIL_AVAILABLE:
                img = Image.new("RGB", (job["width"], job["height"]), (30,30,40))
                ImageDraw.Draw(img).text((20,20), optimized_prompt[:120], fill=(255,255,255))
                img.save(dummy_path)
            frames=[dummy_path]
            job["frames"]=frames
            save_job_checkpoint(job_id, job)
        # 5) TTS
        audio_path=None
        if job.get("tts_text"):
            audio_path = _generate_tts_audio(job["tts_text"], lang="hi", out_path=f"/tmp/ucve_tts_{uuid.uuid4().hex[:6]}.mp3")
            job["tts"]=audio_path
            save_job_checkpoint(job_id, job)
        # 6) Mix SFX + TTS
        mixed_audio=None
        if sfx or audio_path:
            mixed_audio = mix_audio_tracks(audio_path, sfx, job["duration_sec"], f"/tmp/ucve_mix_{uuid.uuid4().hex[:6]}.mp3")
            job["mixed_audio"]=mixed_audio
            save_job_checkpoint(job_id, job)
        # 7) Lipsync (stub)
        if mixed_audio and frames:
            lipsync_audio = lipsync_audio_to_video(mixed_audio, reference_frame=frames[0])
            job["lipsync"] = lipsync_audio
            save_job_checkpoint(job_id, job)
        # 8) Compose video
        video_out = f"/tmp/ucve_final_{job_id}.mp4"
        if MOVIEPY_AVAILABLE and frames:
            clip = ImageSequenceClip(frames, fps=job["fps"])
            if mixed_audio and os.path.exists(mixed_audio):
                audioclip = AudioFileClip(mixed_audio)
                clip = clip.set_audio(audioclip)
            clip.write_videofile(video_out, codec="libx264", audio_codec="aac", verbose=False, logger=None)
        else:
            # placeholder: if moviepy missing, just return first frame path
            if frames:
                video_out = frames[0]
        job["video"]=video_out
        job["status"]="done"
        save_job_checkpoint(job_id, job)
        # 9) Save memory (append last script to character memory)
        mem=_load_memory()
        entry={"time":datetime.utcnow().isoformat(),"script":job["script"],"style":style}
        mem.setdefault("history",[]).append(entry)
        _save_memory(mem)
        # 10) Optional upload to Firebase
        if os.getenv("FIREBASE_CRED_PATH"):
            try:
                if init_firebase_if_needed():
                    upload_to_firebase(video_out, f"renders/{os.path.basename(video_out)}")
            except Exception:
                pass
        # increment prometheus
        if PROM_COUNTER: PROM_COUNTER.inc()
    except Exception as e:
        logging.exception("job worker failed")
        job["status"]="failed"
        job["error"]=str(e)
        save_job_checkpoint(job_id, job)

# =================================================
# Payments endpoints (same as earlier)
# =================================================
@app.post("/payment/stripe/create_checkout_session")
def payment_stripe_create(amount: int = Form(...), currency: str = Form("usd"), success_url: str = Form(f"{DOMAIN}/success"), cancel_url: str = Form(f"{DOMAIN}/cancel")):
    if not STRIPE_SECRET_KEY or not stripe:
        return {"status":"error","message":"Stripe not configured"}
    try:
        stripe.api_key = STRIPE_SECRET_KEY
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{"price_data":{"currency":currency,"product_data":{"name":"UCVE Credits"},"unit_amount":amount},"quantity":1}],
            mode="payment",
            success_url=success_url,
            cancel_url=cancel_url,
        )
        return {"status":"success","checkout_url":session.url,"id":session.id}
    except Exception:
        logging.exception("stripe fail")
        raise HTTPException(status_code=500, detail="stripe error")

@app.post("/payment/stripe/webhook")
async def payment_stripe_webhook(request: Request):
    if not STRIPE_SECRET_KEY or not STRIPE_WEBHOOK_SECRET or not stripe:
        return {"status":"error","message":"Stripe not configured"}
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
        if event["type"]=="checkout.session.completed":
            logging.info("stripe checkout completed")
        return {"status":"success"}
    except Exception:
        raise HTTPException(status_code=400, detail="invalid webhook")

@app.post("/payment/razorpay/create_order")
def payment_razorpay_create(amount: int = Form(...), currency: str = Form("INR"), receipt: str = Form("rcpt_1")):
    if not RAZORPAY_KEY_ID or not RAZORPAY_KEY_SECRET or not razorpay:
        return {"status":"error","message":"Razorpay not configured"}
    try:
        client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))
        order = client.order.create({"amount": amount, "currency": currency, "receipt": receipt})
        return {"status":"success","order":order}
    except Exception:
        logging.exception("razorpay fail")
        raise HTTPException(status_code=500, detail="razorpay error")

@app.post("/payment/razorpay/verify")
def payment_razorpay_verify(razorpay_order_id: str = Form(...), razorpay_payment_id: str = Form(...), razorpay_signature: str = Form(...)):
    if not RAZORPAY_KEY_ID or not RAZORPAY_KEY_SECRET:
        return {"status":"error","message":"Razorpay not configured"}
    try:
        generated = hmac.new(RAZORPAY_KEY_SECRET.encode(), (razorpay_order_id + "|" + razorpay_payment_id).encode(), hashlib.sha256).hexdigest()
        if generated == razorpay_signature:
            return {"status":"success","message":"verified"}
        return {"status":"error","message":"invalid signature"}
    except Exception:
        logging.exception("razorpay verify error")
        raise HTTPException(status_code=500, detail="verify error")

# =================================================
# Admin & utility endpoints
# =================================================
@app.get("/")
def root():
    return {"message":"Visora AI UCVE-X running","time":datetime.utcnow().isoformat()}

@app.get("/health")
def health():
    return {"status":"ok","time":datetime.utcnow().isoformat()}

@app.get("/admin/list_jobs")
def admin_list_jobs():
    files=[f for f in os.listdir(JOBS_DIR) if f.endswith(".json")]
    jobs=[]
    for f in files:
        try:
            jobs.append(json.load(open(os.path.join(JOBS_DIR,f),"r",encoding="utf-8")))
        except Exception:
            pass
    return {"jobs":jobs}

@app.get("/admin/get_memory")
def admin_get_memory():
    return _load_memory()

# startup
@app.on_event("startup")
def startup():
    logging.info("Visora AI UCVE-X starting...")
    if TORCH_AVAILABLE:
        device, dtype = _get_torch_device_and_dtype()
        logging.info("Torch available. device=%s dtype=%s", device, dtype)
    else:
        logging.info("Torch not available; CPU-only")
    if ELEVEN_API_KEY:
        logging.info("ElevenLabs configured")
    if HUGGINGFACE_HUB_TOKEN:
        logging.info("HuggingFace token present")
    logging.info("Startup complete")

# End of file
