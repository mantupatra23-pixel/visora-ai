#!/usr/bin/env python3
"""
Visora Backend - Universal Production Build (Render + Firebase Compatible)
---------------------------------------------------------------------------
Author : Aimantuvya & GPT-5
Version: 1.0
Description:
    - Full-featured backend for Visora AI Video Studio
    - Supports MoviePy rendering, Firebase sync, S3 backup, Razorpay/PayPal
    - Flask + JWT Auth + SQLite/Firestore hybrid
"""

import os, json, uuid, logging, shutil, time, threading, tempfile, subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

from flask import Flask, request, jsonify, send_from_directory, abort
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import jwt

# Optional dependencies
try:
    from moviepy.editor import (
        ImageClip, concatenate_videoclips,
        AudioFileClip, CompositeAudioClip
    )
    MOVIEPY_AVAILABLE = True
except Exception as e:
    print("⚠️ MoviePy not available:", e)
    MOVIEPY_AVAILABLE = False

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except Exception as e:
    print("⚠️ gTTS not available:", e)
    GTTS_AVAILABLE = False

# ==============================================
# 🎬 CINEMATIC VIDEO GENERATION MODULE (AI)
# ==============================================
from elevenlabs import generate, save
from pydub import AudioSegment
import random

def generate_cinematic_video(prompt, duration=10):
    try:
        characters = ["male", "female", "child", "old"]
        voice_type = random.choice(characters)
        voice_map = {
            "male": "21m00Tcm4TlvDq8ikWAM",
            "female": "EXAVITQu4vr4xnSDxMaL",
            "child": "TxGEqnHWrfWFTfGW9XjX",
            "old": "ErXwobaYiN019PkySvjV"
        }

        voice_id = voice_map[voice_type]
        voice_path = f"/tmp/{voice_type}_voice.mp3"

        audio = generate(text=prompt, voice=voice_id, model="eleven_monolingual_v1")
        save(audio, voice_path)

        video_path = "/tmp/bg_cinematic.mp4"
        os.system(f"ffmpeg -y -f lavfi -i color=c=black:s=1080x1920:d={duration} {video_path}")

        bgm_path = "/tmp/bgm.mp3"
        os.system(f"ffmpeg -y -i https://cdn.pixabay.com/audio/2023/03/01/audio_51c2a5d7b3.mp3 -ss 0 -t {duration} {bgm_path}")

        voice = AudioSegment.from_file(voice_path)
        bgm = AudioSegment.from_file(bgm_path).apply_gain(-10)
        final_audio = bgm.overlay(voice)
        final_audio.export("/tmp/final_audio.mp3", format="mp3")

        final_output = "/tmp/final_output.mp4"
        os.system("ffmpeg -y -i /tmp/bg_cinematic.mp4 -i /tmp/final_audio.mp3 -shortest /tmp/final_output.mp4")

        return final_output
    except Exception as e:
        print("❌ Error generating cinematic video:", e)
        return None

# ==============================================
# Wrapper to connect cinematic generator with API
# ==============================================
def build_ai_composed_video(script_text):
    return generate_cinematic_video(script_text, 10)

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
for p in [UPLOAD_DIR, OUTPUT_DIR]:
    p.mkdir(exist_ok=True)

app = Flask("Visora-Backend")
CORS(app)

app.config.update({
    "SECRET_KEY": os.getenv("SECRET_KEY", "visora-secret"),
    "SQLALCHEMY_DATABASE_URI": os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR/'visora.db'}"),
    "SQLALCHEMY_TRACK_MODIFICATIONS": False,
    "JWT_EXPIRY": 48,  # hours
})

db = SQLAlchemy(app)
log = logging.getLogger("visora")
logging.basicConfig(level=logging.INFO)

# ----------------------------------------------------------------------------
# DATABASE MODELS
# ----------------------------------------------------------------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    name = db.Column(db.String(100))
    password = db.Column(db.String(256))
    photo = db.Column(db.String(512))
    plan = db.Column(db.String(50), default="Free")
    credits = db.Column(db.Integer, default=10)
    country = db.Column(db.String(50), default="India")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Video(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_email = db.Column(db.String(120))
    title = db.Column(db.String(255))
    file_path = db.Column(db.String(512))
    status = db.Column(db.String(50), default="pending")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    meta = db.Column(db.Text)

class Template(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120))
    category = db.Column(db.String(100))
    thumbnail = db.Column(db.String(512))
    trending = db.Column(db.Integer, default=0)

class Character(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_email = db.Column(db.String(120))
    name = db.Column(db.String(100))
    photo_path = db.Column(db.String(512))
    voice_path = db.Column(db.String(512))

class Plan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50))
    price = db.Column(db.String(50))
    features = db.Column(db.String(512))

# ----------------------------------------------------------------------------
# UTILS
# ----------------------------------------------------------------------------
def token_create(email: str):
    exp = datetime.utcnow() + timedelta(hours=app.config["JWT_EXPIRY"])
    return jwt.encode({"email": email, "exp": exp}, app.config["SECRET_KEY"], algorithm="HS256")

def token_verify(token: str):
    try:
        return jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
    except Exception:
        return None

def save_file(fs, folder="uploads") -> str:
    filename = secure_name(fs.filename)
    dest = Path(folder) / filename
    fs.save(dest)
    return str(dest)

def secure_name(name: str) -> str:
    name = name.replace(" ", "_")
    uid = uuid.uuid4().hex[:6]
    if "." in name:
        base, ext = name.rsplit(".", 1)
        return f"{base}_{uid}.{ext}"
    return f"{name}_{uid}"

# ====================== AUTH ENDPOINTS ======================

from flask import request, jsonify
import jwt, datetime
from functools import wraps

SECRET_KEY = "your_secret_key_here"

# ---------------- JWT Helper Functions ----------------
def token_create(email):
    payload = {
        "email": email,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(days=3)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def token_verify(token):
    try:
        decoded = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return decoded
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


# ---------------- REGISTER ----------------
@app.route("/register", methods=["POST"])
def register():
    data = request.get_json(force=True)
    email = data.get("email")
    password = data.get("password")
    name = data.get("name", "User")

    if not email or not password:
        return jsonify({"error": "Missing credentials"}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({"error": "User already exists"}), 400

    u = User(email=email, name=name, password=password)
    db.session.add(u)
    db.session.commit()

    token = token_create(email)
    return jsonify({
        "message": "User registered successfully ✅",
        "token": token,
        "user": {"email": u.email, "name": u.name}
    }), 200


# ---------------- LOGIN ----------------
@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(force=True)
    email = data.get("email")
    password = data.get("password")

    u = User.query.filter_by(email=email).first()
    if not u or u.password != password:
        return jsonify({"error": "Invalid credentials"}), 401

    token = token_create(email)
    return jsonify({
        "message": "Login successful 🔐",
        "token": token,
        "user": {"email": u.email, "name": u.name}
    }), 200


# ---------------- PROFILE ----------------
@app.route("/profile", methods=["GET"])
def profile():
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    decoded = token_verify(token)

    if not decoded:
        return jsonify({"error": "Invalid or expired token"}), 401

    u = User.query.filter_by(email=decoded["email"]).first()
    if not u:
        return jsonify({"error": "User not found"}), 404

    return jsonify({
        "email": u.email,
        "name": u.name,
        "plan": getattr(u, "plan", "free"),
        "credits": getattr(u, "credits", 100),
        "country": getattr(u, "country", "Unknown")
    }), 200

# ----------------------------------------------------------------------------
# INIT DATABASE
# ----------------------------------------------------------------------------
with app.app_context():
    db.create_all()
    if not Plan.query.first():
        db.session.add_all([
            Plan(name="Free", price="0", features="480p renders, 5 credits/day"),
            Plan(name="Pro", price="499", features="1080p renders, 50 credits/day"),
            Plan(name="Enterprise", price="999", features="4K renders, unlimited renders")
        ])
        db.session.commit()

print("✅ Visora Backend Part 1 Loaded")

# ----------------------------------------------------------------------------
# NOTE: Continue to Part 2 → Video Rendering, Assistant, Templates, Admin
# ----------------------------------------------------------------------------# -------------------- Part 2: Rendering Queue, Video endpoints, TTS, Assistant, Admin --------------------

import threading
from queue import Queue, Empty
from werkzeug.utils import secure_filename

# Globals for render jobs
render_queue = Queue()
render_jobs: Dict[str, Dict[str, Any]] = {}
RENDER_WORKER_SLEEP = 0.8

# Helper: absolute path from relative
def abs_path(p: str) -> str:
    if not p:
        return p
    p = Path(p)
    if not p.is_absolute():
        p = BASE_DIR / p
    return str(p.resolve())

# Simple movie rendering (moviepy) or fallback mock
def render_images_with_audios(image_paths: List[str], audio_paths: List[str], out_path: str, quality: str = "HD", bg_music: Optional[str] = None):
    """
    - image_paths: list of absolute or relative image file paths
    - audio_paths: list of absolute or relative audio file paths (matching characters)
    - out_path: absolute path to write mp4
    """
    if not MOVIEPY_AVAILABLE:
        # Fallback: create silent mp4 using ffmpeg if available, or raise
        try:
            duration = 3 * max(1, len(image_paths))
            tmp_img = image_paths[0] if image_paths else None
            if not tmp_img:
                raise RuntimeError("No images provided for fallback render")
            # Use ffmpeg to create a slideshow (if ffmpeg installed)
            cmd = [
                "ffmpeg", "-y",
            ]
            # create input list file
            listfile = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
            for img in image_paths:
                listfile.write(f"file '{abs_path(img)}'\n".encode())
                listfile.write(b"duration 2\n")
            listfile.flush(); listfile.close()
            cmd += ["-f", "concat", "-safe", "0", "-i", listfile.name, "-vf", "scale=1280:-2", "-c:v", "libx264", "-r", "24", "-pix_fmt", "yuv420p", out_path]
            subprocess.check_output(cmd, stderr=subprocess.STDOUT)
            try:
                os.unlink(listfile.name)
            except: pass
            return
        except Exception as e:
            raise RuntimeError(f"MoviePy missing and ffmpeg fallback failed: {e}")

    # MoviePy actual flow
    clips = []
    audios = []
    n = min(len(image_paths), len(audio_paths)) if audio_paths else len(image_paths)
    if n == 0:
        raise ValueError("Need at least one image and one audio (or one image to make video).")

    for i in range(n):
        img = abs_path(image_paths[i])
        aud = abs_path(audio_paths[i]) if i < len(audio_paths) else None
        # load audio if exists
        audio_clip = None
        dur = 2.0
        if aud and os.path.exists(aud):
            audio_clip = AudioFileClip(aud)
            dur = max(2.0, getattr(audio_clip, "duration", 2.0))
            audios.append(audio_clip)
        # create lip-sync like image clip
        clip = ImageClip(img).set_duration(dur).resize(width=1280)
        clips.append(clip.set_audio(audio_clip) if audio_clip else clip)
    final = concatenate_videoclips(clips, method="compose")
    # background music
    if bg_music:
        try:
            bg = AudioFileClip(abs_path(bg_music))
            if bg.duration < final.duration:
                # loop bg
                from moviepy.editor import concatenate_audioclips
                times = int(final.duration / bg.duration) + 1
                bg = concatenate_audioclips([bg] * times).subclip(0, final.duration)
            else:
                bg = bg.subclip(0, final.duration)
            bg = bg.volumex(0.12)
            final_audio = CompositeAudioClip([final.audio, bg]) if final.audio else bg
            final = final.set_audio(final_audio)
        except Exception as e:
            log.exception("bg music attach failed: %s", e)
    # write file
    bitrate = "800k"
    if quality and "4k" in quality.lower():
        bitrate = "8000k"
    elif quality and "1080" in quality.lower():
        bitrate = "2500k"
    final.write_videofile(out_path, fps=24, codec="libx264", audio_codec="aac", bitrate=bitrate)
    final.close()
    for a in audios:
        try: a.close()
        except: pass

# Render worker thread
def worker_loop():
    log.info("Render worker started")
    while True:
        try:
            job = render_queue.get(timeout=1)
        except Empty:
            time.sleep(RENDER_WORKER_SLEEP)
            continue
        job_id = job.get("job_id")
        render_jobs[job_id]["status"] = "processing"
        try:
            images = job.get("images", [])
            audios = job.get("audios", [])
            bg = job.get("bg")
            quality = job.get("quality","HD")
            out_name = f"{job_id}.mp4"
            out_abs = str(OUTPUT_DIR / out_name)
            render_images_with_audios(images, audios, out_abs, quality=quality, bg_music=bg)
            render_jobs[job_id]["status"] = "done"
            render_jobs[job_id]["output"] = str(Path("outputs") / out_name)
            # Save DB record if video record provided
            vid_id = job.get("video_db_id")
            if vid_id:
                with app.app_context():
                    v = Video.query.get(vid_id)
                    if v:
                        v.file_path = render_jobs[job_id]["output"]
                        v.status = "done"
                        db.session.commit()
            log.info("Render job done: %s", job_id)
        except Exception as e:
            log.exception("Render failed for %s: %s", job_id, e)
            render_jobs[job_id]["status"] = "failed"
            render_jobs[job_id]["error"] = str(e)
            # mark DB if present
            vid_id = job.get("video_db_id")
            if vid_id:
                with app.app_context():
                    v = Video.query.get(vid_id)
                    if v:
                        v.status = "failed"
                        db.session.commit()
        finally:
            try:
                render_queue.task_done()
            except:
                pass

worker_thread = threading.Thread(target=worker_loop, daemon=True)
worker_thread.start()

# -------------------- API: generate_video --------------------
@app.route("/generate_video", methods=["POST"])
def generate_video():
    """
    Accepts multipart/form-data:
    - token (header) or user_email (form)
    - title
    - template
    - quality
    - bg_music (choice) or bg_music_file
    - character_images[] (multiple)
    - character_audios[] (optional multiple)
    - script (optional)
    """
    # auth fallback
    token = request.headers.get("Authorization","").replace("Bearer ","")
    user_email = None
    if token:
        decoded = token_verify(token)
        user_email = decoded.get("email") if decoded else None
    if not user_email:
        user_email = request.form.get("user_email","demo@visora.com")

    title = request.form.get("title") or f"Visora_{uuid.uuid4().hex[:8]}"
    template = request.form.get("template","Default")
    quality = request.form.get("quality","HD")
    script = request.form.get("script","")
    # handle bg music upload
    bg_music_rel = None
    if "bg_music_file" in request.files:
        f = request.files["bg_music_file"]
        fname = secure_filename(f.filename)
        dest = UPLOAD_DIR / "bg_music"
        dest.mkdir(parents=True, exist_ok=True)
        pth = dest / f"{uuid.uuid4().hex}_{fname}"
        f.save(pth)
        bg_music_rel = str(pth)
    elif request.form.get("bg_music"):
        # preset name - left as future mapping
        bg_music_rel = request.form.get("bg_music")

    # images
    images = []
    if "character_images" in request.files:
        files = request.files.getlist("character_images")
        for f in files:
            fname = secure_filename(f.filename)
            destdir = UPLOAD_DIR / "images"
            destdir.mkdir(parents=True, exist_ok=True)
            dest = destdir / f"{uuid.uuid4().hex}_{fname}"
            f.save(dest)
            images.append(str(dest))
    # audios
    audios = []
    if "character_audios" in request.files:
        files = request.files.getlist("character_audios")
        for f in files:
            fname = secure_filename(f.filename)
            destdir = UPLOAD_DIR / "audios"
            destdir.mkdir(parents=True, exist_ok=True)
            dest = destdir / f"{uuid.uuid4().hex}_{fname}"
            f.save(dest)
            audios.append(str(dest))

    # create DB record
    v = Video(user_email=user_email, title=title, status="queued", meta=json.dumps({"template": template, "script": script}))
    db.session.add(v); db.session.commit()

    # enqueue
    job_id = uuid.uuid4().hex
    render_jobs[job_id] = {"job_id": job_id, "status": "queued", "created_at": datetime.utcnow().isoformat(), "title": title}
    job_payload = {
        "job_id": job_id,
        "video_db_id": v.id,
        "images": images,
        "audios": audios,
        "bg": bg_music_rel,
        "quality": quality
    }
    render_queue.put(job_payload)
    render_jobs[job_id]["queued_at"] = datetime.utcnow().isoformat()

    return jsonify({"message":"enqueued", "job_id": job_id, "video_id": v.id})

# -------------------- API: job_status --------------------
@app.route("/job_status", methods=["GET"])
def job_status():
    job_id = request.args.get("job_id")
    if not job_id:
        return jsonify({"error":"job_id required"}), 400
    j = render_jobs.get(job_id)
    if not j:
        return jsonify({"error":"not found"}), 404
    return jsonify(j)

# -------------------- Gallery list --------------------
@app.route("/gallery", methods=["GET"])
def gallery_list():
    user = request.args.get("user_email","demo@visora.com")
    vids = Video.query.filter_by(user_email=user).order_by(Video.created_at.desc()).all()
    out = []
    for v in vids:
        meta = {}
        try:
            meta = json.loads(v.meta) if v.meta else {}
        except:
            meta = {}
        out.append({"id": v.id, "title": v.title, "status": v.status, "file": v.file_path, "meta": meta, "created_at": v.created_at.isoformat()})
    return jsonify(out)

# -------------------- Voice upload & preview --------------------
@app.route("/upload_voice", methods=["POST"])
def upload_voice_file():
    token = request.headers.get("Authorization","").replace("Bearer ","")
    user_email = request.form.get("user_email","demo@visora.com")
    if "file" not in request.files:
        return jsonify({"error":"no file"}), 400
    f = request.files["file"]
    fname = secure_filename(f.filename)
    destdir = UPLOAD_DIR / "voices"
    destdir.mkdir(parents=True, exist_ok=True)
    dest = destdir / f"{uuid.uuid4().hex}_{fname}"
    f.save(dest)
    # Optionally associate with Character
    char_id = request.form.get("character_id")
    if char_id:
        with app.app_context():
            ch = Character.query.get(int(char_id))
            if ch:
                ch.voice_path = str(dest)
                db.session.commit()
    return jsonify({"saved": str(dest)})

@app.route("/preview_tts", methods=["POST"])
def preview_tts():
    if not GTTS_AVAILABLE:
        return jsonify({"error":"gTTS not available"}), 500
    text = request.form.get("text", "Hello from Visora")
    lang = request.form.get("lang","hi")
    try:
        uid = uuid.uuid4().hex
        out = UPLOAD_DIR / "tmp" / f"tts_{uid}.mp3"
        out.parent.mkdir(parents=True, exist_ok=True)
        gTTS(text, lang=lang).save(str(out))
        return jsonify({"audio": str(out)})
    except Exception as e:
        log.exception("TTS failed")
        return jsonify({"error":"tts failed","details":str(e)}),500

# -------------------- AI Assistant (OpenAI integration ready) --------------------
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # default; you can set GPT-5 model name

def assistant_generate_local(prompt: str, tone: str = "helpful"):
    # Simple local heuristic fallback when OpenAI key missing
    if tone == "funny":
        return f"😂 Funny start idea: \"Guess what... {prompt[:80]}\""
    if tone == "motivational":
        return f"🔥 Motivational hook: Start with 'Never give up...' then mention {prompt[:80]}"
    return f"Try a punchy hook: '{prompt[:60]}...' and close with a one-line CTA."

@app.route("/assistant", methods=["POST"])
def assistant_api():
    data = request.get_json() or {}
    prompt = data.get("query","")
    tone = data.get("tone","helpful")
    lang = data.get("lang","en")
    # If OPENAI_KEY exists, call OpenAI
    if OPENAI_KEY:
        try:
            import requests
            headers = {"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"}
            body = {
                "model": OPENAI_MODEL,
                "input": f"Tone: {tone}\nPrompt: {prompt}"
            }
            # Using Chat Completions or Responses endpoint depends on OpenAI setup
            resp = requests.post("https://api.openai.com/v1/responses", headers=headers, json=body, timeout=30)
            if resp.status_code == 200:
                j = resp.json()
                # try to extract reply text
                text = ""
                if "output" in j and isinstance(j["output"], list):
                    pieces = [p.get("content","") if isinstance(p, dict) else str(p) for p in j["output"]]
                    text = " ".join(pieces)
                elif "choices" in j and j["choices"]:
                    text = j["choices"][0].get("message", {}).get("content", "")
                else:
                    text = str(j)
                # optionally gTTS audio
                audio_url = None
                if GTTS_AVAILABLE:
                    try:
                        uid = uuid.uuid4().hex
                        out = UPLOAD_DIR / "assistant_audio" / f"assistant_{uid}.mp3"
                        out.parent.mkdir(parents=True, exist_ok=True)
                        # choose language param (basic)
                        gTTS(text, lang=lang).save(str(out))
                        audio_url = str(out)
                    except Exception:
                        audio_url = None
                return jsonify({"reply": text, "audio": audio_url})
            else:
                log.warning("OpenAI error %s %s", resp.status_code, resp.text)
                # fallback
                return jsonify({"reply": assistant_generate_local(prompt, tone)}), 200
        except Exception as e:
            log.exception("OpenAI call failed")
            return jsonify({"reply": assistant_generate_local(prompt, tone), "error": str(e)}), 200
    else:
        # local fallback
        return jsonify({"reply": assistant_generate_local(prompt, tone)}), 200

# -------------------- Templates & Admin endpoints --------------------
@app.route("/templates", methods=["GET"])
def api_templates():
    tpls = Template.query.order_by(Template.trending.desc()).all()
    out = [{"id": t.id, "name": t.name, "category": t.category, "thumbnail": t.thumbnail, "trending": t.trending} for t in tpls]
    return jsonify(out)

@app.route("/admin/templates", methods=["POST"])
def admin_create_template():
    data = request.get_json() or {}
    name = data.get("name"); cat = data.get("category","General"); thumb = data.get("thumbnail","")
    t = Template(name=name, category=cat, thumbnail=thumb)
    db.session.add(t); db.session.commit()
    return jsonify({"message":"created","id": t.id})

@app.route("/admin/templates/<int:tid>", methods=["PUT","DELETE"])
def admin_update_template(tid):
    t = Template.query.get(tid)
    if not t:
        return jsonify({"error":"not found"}), 404
    if request.method == "PUT":
        data = request.get_json() or {}
        t.name = data.get("name", t.name)
        t.category = data.get("category", t.category)
        t.thumbnail = data.get("thumbnail", t.thumbnail)
        t.trending = data.get("trending", t.trending)
        db.session.commit()
        return jsonify({"message":"updated"})
    else:
        db.session.delete(t); db.session.commit()
        return jsonify({"message":"deleted"})

# Admin: user management
@app.route("/admin/users", methods=["GET"])
def admin_list_users():
    users = User.query.all()
    out = [{"email": u.email, "name": u.name, "plan": u.plan, "credits": u.credits} for u in users]
    return jsonify(out)

@app.route("/admin/users/<string:email>/credits", methods=["POST"])
def admin_modify_credits(email):
    data = request.get_json() or {}
    delta = int(data.get("delta",0))
    u = User.query.filter_by(email=email).first()
    if not u:
        return jsonify({"error":"not found"}), 404
    u.credits = max(0, u.credits + delta)
    db.session.commit()
    return jsonify({"message":"ok","credits": u.credits})

# Simple analytics
@app.route("/admin/analytics", methods=["GET"])
def admin_analytics():
    total_users = User.query.count()
    total_videos = Video.query.count()
    pending = Video.query.filter_by(status="queued").count()
    done = Video.query.filter_by(status="done").count()
    return jsonify({"users": total_users, "videos": total_videos, "pending": pending, "done": done})

print("✅ Visora Backend Part 2 Loaded")# -------------------- Part 3: Payments, Cloud Sync, Conversion, Scheduler, Deployment helpers --------------------

import os
import subprocess
import atexit
import boto3
from botocore.exceptions import BotoCoreError, ClientError

# Optional: firebase admin
try:
    import firebase_admin
    from firebase_admin import credentials, storage as fb_storage
    FIREBASE_AVAILABLE = True
except Exception as e:
    FIREBASE_AVAILABLE = False

# Optional: Razorpay
try:
    import razorpay
    RAZORPAY_AVAILABLE = True
except Exception:
    RAZORPAY_AVAILABLE = False

# PayPal (we will use requests)
import requests

# ---------- Config from env ----------
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
AWS_S3_REGION = os.getenv("AWS_S3_REGION", "us-east-1")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

FIREBASE_CRED_JSON = os.getenv("FIREBASE_CRED_JSON")  # path to service account json (optional)
FIREBASE_BUCKET = os.getenv("FIREBASE_BUCKET")

PAYPAL_CLIENT_ID = os.getenv("PAYPAL_CLIENT_ID")
PAYPAL_SECRET = os.getenv("PAYPAL_SECRET")
PAYPAL_SANDBOX = os.getenv("PAYPAL_SANDBOX", "1") == "1"

RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET")

# initialize S3 client if possible
s3_client = None
if AWS_ACCESS_KEY and AWS_SECRET_KEY and AWS_S3_BUCKET:
    try:
        s3_client = boto3.client("s3", region_name=AWS_S3_REGION, aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
        log.info("S3 client initialized for bucket %s", AWS_S3_BUCKET)
    except Exception as e:
        log.exception("Failed to init S3 client: %s", e)
        s3_client = None

# firebase init
if FIREBASE_AVAILABLE and FIREBASE_CRED_JSON:
    try:
        cred = credentials.Certificate(FIREBASE_CRED_JSON)
        firebase_admin.initialize_app(cred, {"storageBucket": FIREBASE_BUCKET})
        log.info("Firebase admin initialized for bucket %s", FIREBASE_BUCKET)
    except Exception as e:
        log.exception("Firebase init failed: %s", e)
        FIREBASE_AVAILABLE = False

# razorpay init
razor_client = None
if RAZORPAY_AVAILABLE and RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET:
    try:
        razor_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))
        log.info("Razorpay client initialized")
    except Exception as e:
        log.exception("Razorpay init failed: %s", e)
        razor_client = None

# ---------- Helper: upload to S3 ----------
def upload_file_to_s3(local_path: str, key: str) -> Optional[str]:
    if not s3_client:
        log.warning("S3 not configured")
        return None
    try:
        s3_client.upload_file(local_path, AWS_S3_BUCKET, key, ExtraArgs={"ACL":"public-read"})
        url = f"https://{AWS_S3_BUCKET}.s3.{AWS_S3_REGION}.amazonaws.com/{key}"
        return url
    except Exception as e:
        log.exception("S3 upload failed: %s", e)
        return None

# ---------- Helper: upload to Firebase ----------
def upload_file_to_firebase(local_path: str, dest_name: str) -> Optional[str]:
    if not FIREBASE_AVAILABLE:
        log.warning("Firebase not available")
        return None
    try:
        bucket = fb_storage.bucket()
        blob = bucket.blob(dest_name)
        blob.upload_from_filename(local_path)
        # make public (optional)
        try:
            blob.make_public()
            return blob.public_url
        except Exception:
            return blob.generate_signed_url(datetime.timedelta(days=365))
    except Exception as e:
        log.exception("Firebase upload failed: %s", e)
        return None

# ---------- Payments endpoints (Razorpay + PayPal) ----------
@app.route("/create_razorpay_order", methods=["POST"])
def create_razorpay_order_endpoint():
    data = request.get_json() or {}
    amount = int(data.get("amount", 499))  # rupees
    currency = data.get("currency", "INR")
    receipt = f"order_{uuid.uuid4().hex[:8]}"
    if razor_client:
        try:
            order = razor_client.order.create({"amount": amount * 100, "currency": currency, "receipt": receipt})
            return jsonify({"order": order})
        except Exception as e:
            log.exception("Razorpay create order failed")
            return jsonify({"error": "razorpay_error", "details": str(e)}), 500
    else:
        # sandbox mock
        return jsonify({"order": {"id": "mock_"+receipt, "amount": amount * 100, "currency": currency}})

# PayPal token helper
def paypal_get_token():
    base = "https://api-m.sandbox.paypal.com" if PAYPAL_SANDBOX else "https://api-m.paypal.com"
    try:
        resp = requests.post(f"{base}/v1/oauth2/token", auth=(PAYPAL_CLIENT_ID, PAYPAL_SECRET), headers={"Accept":"application/json"}, data={"grant_type":"client_credentials"}, timeout=10)
        if resp.status_code == 200:
            return resp.json().get("access_token")
    except Exception as e:
        log.exception("PayPal token failed: %s", e)
    return None

@app.route("/create_paypal_order", methods=["POST"])
def create_paypal_order_endpoint():
    data = request.get_json() or {}
    amount = data.get("amount", "4.99")
    currency = data.get("currency", "USD")
    if not PAYPAL_CLIENT_ID or not PAYPAL_SECRET:
        return jsonify({"error": "paypal_not_configured", "sandbox": True}), 501
    token = paypal_get_token()
    if not token:
        return jsonify({"error":"paypal_token_failed"}), 500
    base = "https://api-m.sandbox.paypal.com" if PAYPAL_SANDBOX else "https://api-m.paypal.com"
    body = {
        "intent": "CAPTURE",
        "purchase_units": [{"amount": {"currency_code": currency, "value": str(amount)}}]
    }
    resp = requests.post(f"{base}/v2/checkout/orders", headers={"Authorization": f"Bearer {token}", "Content-Type":"application/json"}, json=body, timeout=15)
    if resp.status_code in (200,201):
        return jsonify(resp.json())
    else:
        log.warning("PayPal create order failed: %s %s", resp.status_code, resp.text)
        return jsonify({"error":"paypal_create_failed","details":resp.text}), 500

# ---------- FFmpeg based conversion/compression endpoints ----------
def ffmpeg_installed() -> bool:
    try:
        subprocess.check_output(["ffmpeg", "-version"])
        return True
    except Exception:
        return False

FFMPEG_AVAILABLE = ffmpeg_installed()

@app.route("/convert/video_to_audio", methods=["POST"])
def video_to_audio():
    if "file" not in request.files:
        return jsonify({"error":"no file"}), 400
    f = request.files["file"]
    fname = secure_filename(f.filename)
    tempdir = Path("/tmp") / "visora"
    tempdir.mkdir(parents=True, exist_ok=True)
    fp = tempdir / f"{uuid.uuid4().hex}_{fname}"
    f.save(fp)
    out = tempdir / f"{fp.stem}.mp3"
    try:
        if not FFMPEG_AVAILABLE:
            return jsonify({"error":"ffmpeg not available"}), 500
        cmd = ["ffmpeg","-y","-i", str(fp), "-vn","-acodec","libmp3lame","-q:a","2", str(out)]
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return jsonify({"audio": str(out)})
    except subprocess.CalledProcessError as e:
        log.exception("ffmpeg convert failed: %s", e.output if hasattr(e,"output") else e)
        return jsonify({"error":"convert_failed"}), 500

@app.route("/compress/video", methods=["POST"])
def compress_video():
    # accepts file upload, target_bitrate optional (e.g., 800k)
    if "file" not in request.files:
        return jsonify({"error":"no file"}), 400
    target = request.form.get("target_bitrate","800k")
    f = request.files["file"]
    fname = secure_filename(f.filename)
    tempdir = Path("/tmp") / "visora"
    tempdir.mkdir(parents=True, exist_ok=True)
    fp = tempdir / f"{uuid.uuid4().hex}_{fname}"
    f.save(fp)
    out = tempdir / f"{fp.stem}_compressed.mp4"
    try:
        if not FFMPEG_AVAILABLE:
            return jsonify({"error":"ffmpeg not available"}), 500
        cmd = ["ffmpeg","-y","-i", str(fp), "-b:v", target, "-bufsize", target, "-maxrate", target, str(out)]
        subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=120)
        return jsonify({"output": str(out)})
    except Exception as e:
        log.exception("compress failed: %s", e)
        return jsonify({"error":"compress_failed","details":str(e)}), 500

# ---------- Audio -> Text (Whisper stub) ----------
@app.route("/audio_to_text", methods=["POST"])
def audio_to_text():
    # This is a stub: integrate Whisper or OpenAI Whisper API
    if "file" not in request.files:
        return jsonify({"error":"no file"}), 400
    # Save file and return mock transcription
    f = request.files["file"]
    fname = secure_filename(f.filename)
    fp = UPLOAD_DIR / "audio" / f"{uuid.uuid4().hex}_{fname}"
    fp.parent.mkdir(parents=True, exist_ok=True)
    f.save(fp)
    # In production: call Whisper or OpenAI transcription and return real result
    return jsonify({"transcript": f"(mock) Transcription for {fname}", "file": str(fp)})

# ---------- Scheduler: cleanup tmp files and optional upload sync ----------
CLEANUP_INTERVAL = int(os.getenv("CLEANUP_INTERVAL_SEC", 3600))  # default 1 hour
FILE_RETENTION_DAYS = int(os.getenv("FILE_RETENTION_DAYS", 7))

def cleanup_tmp_and_old_outputs():
    log.info("Cleanup thread started")
    while True:
        try:
            # tmp cleanup
            tmpdir = Path("/tmp") / "visora"
            if tmpdir.exists():
                for f in tmpdir.iterdir():
                    try:
                        if f.is_file() and (time.time() - f.stat().st_mtime) > (3600 * 24 * FILE_RETENTION_DAYS):
                            f.unlink()
                    except Exception:
                        pass
            # outputs cleanup (optionally upload to S3 or Firebase before deletion)
            for out in OUTPUT_DIR.iterdir():
                try:
                    if out.is_file():
                        age_days = (time.time() - out.stat().st_mtime) / (3600*24)
                        if age_days > FILE_RETENTION_DAYS:
                            # optional upload before deletion
                            key = f"backups/{out.name}"
                            if s3_client:
                                upload_file_to_s3(str(out), key)
                            elif FIREBASE_AVAILABLE:
                                upload_file_to_firebase(str(out), f"backups/{out.name}")
                            out.unlink()
                except Exception:
                    pass
        except Exception as e:
            log.exception("Cleanup iteration failed: %s", e)
        time.sleep(CLEANUP_INTERVAL)

cleanup_thread = threading.Thread(target=cleanup_tmp_and_old_outputs, daemon=True)
cleanup_thread.start()

# Ensure graceful exit
def on_exit():
    log.info("Shutting down: waiting render queue to finish")
    try:
        render_queue.join(timeout=2)
    except Exception:
        pass

atexit.register(on_exit)

print("✅ Visora Backend Part 3 Loaded")

# -------------------- 3-Layer Voice System (ElevenLabs + OpenAI + gTTS) --------------------
import base64, requests, openai

ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY", "")
ELEVEN_VOICE_ID = os.getenv("ELEVEN_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
openai.api_key = OPENAI_API_KEY

# Character → Voice Map
CHARACTER_VOICES = {
    "C1": {"type": "male", "voice": "Adam"},
    "C2": {"type": "female", "voice": "Rachel"},
    "C3": {"type": "child", "voice": "Charlie"},
    "C4": {"type": "old", "voice": "George"},
}

def generate_character_voice(character_tag, text):
    """
    Generate cinematic voice per character with fallback (ElevenLabs → OpenAI → gTTS)
    """
    if not text.strip():
        return None

    voice_data = CHARACTER_VOICES.get(character_tag, {"type": "neutral", "voice": "Rachel"})
    voice_type = voice_data["type"]
    prefix = f"{character_tag}_{voice_type}"

    out_dir = UPLOAD_FOLDER / "ai_voices"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{prefix}_{uuid.uuid4().hex}.mp3"

    # 1️⃣ ElevenLabs Layer
    if ELEVEN_API_KEY:
        try:
            voice_id = {
                "male": "pNInz6obpgDQGcFmaJgB",
                "female": "21m00Tcm4TlvDq8ikWAM",
                "child": "EXAVITQu4vr4xnSDxMaL",
                "old": "TxGEqnHWrfWFTfGW9XjX",
            }.get(voice_type, ELEVEN_VOICE_ID)

            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            headers = {
                "xi-api-key": ELEVEN_API_KEY,
                "Accept": "audio/mpeg",
                "Content-Type": "application/json"
            }
            payload = {"text": text, "model_id": "eleven_multilingual_v2"}
            r = requests.post(url, headers=headers, json=payload, timeout=60)
            if r.status_code == 200:
                with open(out_file, "wb") as f:
                    f.write(r.content)
                log.info(f"🎙️ ElevenLabs {voice_type} voice generated for {character_tag}")
                return str(out_file)
        except Exception as e:
            log.warning("ElevenLabs voice failed: %s", e)

    # 2️⃣ OpenAI Layer
    if OPENAI_API_KEY:
        try:
            response = openai.audio.speech.create(
                model="gpt-4o-mini-tts",
                voice="alloy" if voice_type in ["male", "old"] else "verse",
                input=text
            )
            with open(out_file, "wb") as f:
                f.write(response.audio)
            log.info(f"🧩 OpenAI TTS generated for {character_tag} ({voice_type})")
            return str(out_file)
        except Exception as e:
            log.warning("OpenAI TTS failed: %s", e)

    # 3️⃣ gTTS Fallback Layer
    try:
        from gtts import gTTS
        gTTS(text, lang="hi").save(str(out_file))
        log.info(f"🔊 gTTS fallback voice generated for {character_tag}")
        return str(out_file)
    except Exception as e:
        log.warning("gTTS fallback failed: %s", e)
    return None

# -------------------- Auto Character Voice Detection System --------------------
import re

def detect_characters_from_script(script_text: str) -> list:
    """
    Parse script lines and auto-assign voice type based on names / tone / gender hints.
    """
    lines = script_text.splitlines()
    detected = []
    voice_types = ["male", "female", "child", "old"]
    assigned = {}

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Try to detect name: "Aman: Hello"
        match = re.match(r"^([A-Za-z]+):\s*(.*)", line)
        if match:
            name, text = match.groups()
        else:
            # Default fallback
            name, text = "Unknown", line

        # Assign voice if not already
        if name not in assigned:
            if any(x in name.lower() for x in ["riya", "anita", "sita", "she", "her"]):
                assigned[name] = "female"
            elif any(x in name.lower() for x in ["kid", "child", "boy", "girl"]):
                assigned[name] = "child"
            elif any(x in name.lower() for x in ["grand", "old", "baba", "dada", "amma"]):
                assigned[name] = "old"
            else:
                assigned[name] = "male"

        detected.append({
            "character": name,
            "voice_type": assigned[name],
            "text": text
        })

    log.info(f"🧠 Auto detected characters: {json.dumps(assigned)}")
    return detected


def process_script_auto(script_text: str):
    """
    Automatically process script → detect characters → generate voice per line.
    Returns list of generated voice file paths.
    """
    segments = detect_characters_from_script(script_text)
    voice_files = []
    for seg in segments:
        voice_path = generate_character_voice(seg["character"], seg["text"])
        if voice_path:
            voice_files.append(voice_path)
    return voice_files

# -------------------- Auto Avatar Fetcher + Character Visual Scene Builder --------------------
import urllib.parse

UNSPLASH_API_KEY = os.getenv("UNSPLASH_API_KEY", "")
UNSPLASH_FALLBACKS = {
    "male": "https://images.unsplash.com/photo-1603415526960-f7e0328c63b1",
    "female": "https://images.unsplash.com/photo-1534528741775-53994a69daeb",
    "child": "https://images.unsplash.com/photo-1503457574463-6b2bca43b5cb",
    "old": "https://images.unsplash.com/photo-1529626455594-4ff0802cfb7e"
}

def fetch_avatar_for_character(name: str, voice_type: str) -> str:
    """
    Fetch an avatar image for given character using Unsplash API or fallback.
    """
    try:
        query = urllib.parse.quote_plus(name)
        if UNSPLASH_API_KEY:
            url = f"https://api.unsplash.com/search/photos?query={query}&client_id={UNSPLASH_API_KEY}&orientation=squarish&per_page=1"
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if data["results"]:
                    return data["results"][0]["urls"]["regular"]
        return UNSPLASH_FALLBACKS.get(voice_type, UNSPLASH_FALLBACKS["male"])
    except Exception as e:
        log.warning(f"Avatar fetch failed for {name}: {e}")
        return UNSPLASH_FALLBACKS.get(voice_type, UNSPLASH_FALLBACKS["male"])


def build_cinematic_scenes(script_text: str):
    """
    Build cinematic scenes with avatar + voice per line.
    Returns (image_urls, voice_files)
    """
    segments = detect_characters_from_script(script_text)
    image_urls = []
    voice_files = []

    for seg in segments:
        name = seg["character"]
        voice_type = seg["voice_type"]
        text = seg["text"]
        avatar_url = fetch_avatar_for_character(name, voice_type)
        image_urls.append(avatar_url)

        voice_path = generate_character_voice(seg["character"], text)
        if voice_path:
            voice_files.append(voice_path)

        log.info(f"🎭 Scene built for {name} ({voice_type}) → voice: {voice_path}, avatar: {avatar_url}")

    return image_urls, voice_files

# ---------------- Emotion-Based Cinematic Tone Enhancement ----------------
import textblob
from moviepy.editor import (
    vfx,
    ImageClip,
    concatenate_videoclips,
    AudioFileClip
)
from pathlib import Path
import uuid
import logging as log

# Make sure the output folder exists
OUTPUT_FOLDER = Path("outputs")
OUTPUT_FOLDER.mkdir(exist_ok=True)

# 💡 Detect Emotion from Script Text
def analyze_emotion_from_text(text: str) -> str:
    """
    Detect emotion (happy, sad, angry, excited, neutral) from text using TextBlob.
    """
    try:
        blob = textblob.TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0.5:
            return "excited"
        elif polarity > 0.1:
            return "happy"
        elif polarity < -0.4:
            return "angry"
        elif polarity < 0:
            return "sad"
        else:
            return "neutral"
    except Exception as e:
        log.warning(f"Emotion detection failed: {e}")
        return "neutral"

# 🎨 Apply Emotion-Based Visual Filter
def apply_emotion_filter(clip, emotion: str):
    """
    Apply color tone, zoom, and brightness based on detected emotion.
    """
    try:
        if emotion == "happy":
            clip = clip.fx(vfx.colorx, 1.2)
        elif emotion == "excited":
            clip = clip.fx(vfx.colorx, 1.4).fx(vfx.speedx, 1.1)
        elif emotion == "sad":
            clip = clip.fx(vfx.colorx, 0.7).fx(vfx.lum_contrast, 0.8, 0.9, 128)
        elif emotion == "angry":
            clip = clip.fx(vfx.colorx, 0.9).fx(vfx.lum_contrast, 1.3, 1.2, 128)
        else:
            clip = clip.fx(vfx.colorx, 1.0)
        return clip
    except Exception as e:
        log.warning(f"Emotion filter failed: {e}")
        return clip

# 🎬 Build Full Emotion-Cinematic Video
def build_emotion_cinematic_video(script_text: str) -> str:
    """
    Combine emotion detection + avatar + voice to build cinematic video.
    """
    try:
        segments = detect_characters_from_script(script_text)
        clips = []

        for seg in segments:
            name = seg.get("character", "Unknown")
            voice_type = seg.get("voice_type", "neutral")
            text = seg.get("text", "")

            emotion = analyze_emotion_from_text(text)
            avatar_url = fetch_avatar_for_character(name, voice_type)

            log.info(f"🎞️ Scene: {name} ({emotion}) -> {text}")

            try:
                # 🖼️ Create character clip + 🎤 voice
                img_clip = ImageClip(avatar_url).resize((720, 1280))
                voice_path = generate_character_voice(name, voice_type, text)
                audio_clip = AudioFileClip(voice_path)
                dur = max(audio_clip.duration, 3)

                # 🎥 Apply cinematic & emotion filters
                img_clip = cinematic_motion(img_clip)
                img_clip = apply_emotion_filter(img_clip, emotion)

                final_clip = img_clip.set_duration(dur).set_audio(audio_clip)
                clips.append(final_clip)
            except Exception as e:
                log.warning(f"Scene generation failed for {name}: {e}")

        if clips:
            final_video = concatenate_videoclips(clips, method="compose")
            output_path = OUTPUT_FOLDER / f"emotion_story_{uuid.uuid4().hex[:8]}.mp4"
            final_video.write_videofile(str(output_path), fps=24)
            log.info(f"✅ Emotion-based cinematic video created: {output_path}")
            return str(output_path)
        else:
            log.warning("⚠️ No clips generated.")
            return None

    except Exception as e:
        log.error(f"❌ Error in build_emotion_cinematic_video: {e}")
        return None

# =========================
# UNIVERSAL CHARACTER VISUAL ENGINE (UCVE)
# =========================
import os
import uuid
import tempfile
import shutil
import subprocess
from typing import List, Tuple

# --- Character models (avatar clips or placeholder images)
# Replace avatar paths with your cloud URLs or uploads as needed.
CHARACTER_MODELS = {
    "man":     {"avatar":"assets/avatars/man_loop.mp4",     "voice":"male"},
    "woman":   {"avatar":"assets/avatars/woman_loop.mp4",   "voice":"female"},
    "child":   {"avatar":"assets/avatars/child_loop.mp4",   "voice":"child"},
    "old_man": {"avatar":"assets/avatars/oldman_loop.mp4",  "voice":"old"},
    "god":     {"avatar":"assets/avatars/god_loop.mp4",     "voice":"divine"},
    "tiger":   {"avatar":"assets/avatars/tiger_loop.mp4",   "voice":"tiger"},
    "monkey":  {"avatar":"assets/avatars/monkey_loop.mp4",  "voice":"monkey"},
    "lion":    {"avatar":"assets/avatars/lion_loop.mp4",    "voice":"lion"},
    "elephant":{"avatar":"assets/avatars/elephant_loop.mp4","voice":"elephant"},
    "fox":     {"avatar":"assets/avatars/fox_loop.mp4",     "voice":"fox"},
    "robot":   {"avatar":"assets/avatars/robot_loop.mp4",   "voice":"robot"},
    "alien":   {"avatar":"assets/avatars/alien_loop.mp4",   "voice":"alien"},
    "narrator":{"avatar":"assets/avatars/narrator_loop.mp4","voice":"male"},
}

# ============================================
# 🤖 AUTO CHARACTER SCENE GENERATOR (ACSG)
# ============================================

import re
import random

def auto_detect_character(line: str) -> str:
    """
    Automatically detect speaker character type based on the text content.
    """
    line = line.lower()
    if any(word in line for word in ["roar", "hunt", "jungle", "sher", "tiger"]):
        return "tiger"
    elif any(word in line for word in ["bandar", "monkey", "tree", "jump"]):
        return "monkey"
    elif any(word in line for word in ["raj", "king", "yudh", "queen", "mahal"]):
        return random.choice(["man", "woman"])
    elif any(word in line for word in ["dev", "god", "bhagwan", "temple", "light"]):
        return "god"
    elif any(word in line for word in ["child", "baby", "bacha", "school"]):
        return "child"
    elif any(word in line for word in ["woman", "girl", "ladki", "maa"]):
        return "woman"
    elif any(word in line for word in ["old", "baba", "grandpa", "dada", "nana"]):
        return "old_man"
    elif any(word in line for word in ["machine", "robot", "ai", "system"]):
        return "robot"
    elif any(word in line for word in ["alien", "space", "planet", "mars"]):
        return "alien"
    elif any(word in line for word in ["elephant", "haathi", "trunk"]):
        return "elephant"
    elif any(word in line for word in ["lion", "roar", "king of jungle"]):
        return "lion"
    elif any(word in line for word in ["fox", "chalak", "smart"]):
        return "fox"
    else:
        return random.choice(["man", "woman"])  # Default fallback


def generate_auto_scene_script(script_text: str):
    """
    Automatically generate dialogue mapping for all detected characters.
    Example Input:
        "Jungle me ek sher rehta tha. Usne bola main jungle ka raja hoon!"
    Output:
        [
            ("tiger", "Main jungle ka raja hoon!"),
            ("monkey", "Mujhe bhi kursi chahiye!")
        ]
    """
    sentences = re.split(r'[.!?]', script_text)
    dialogues = []

    for line in sentences:
        line = line.strip()
        if not line:
            continue
        character = auto_detect_character(line)
        dialogues.append((character, line))

    print(f"🎭 Auto-detected characters in script: {[c for c, _ in dialogues]}")
    return dialogues

# ============================================
# 🌈 THEME-BASED CINEMATIC BACKGROUND SYSTEM
# ============================================

def detect_story_theme(script_text: str) -> str:
    """
    Detects the main theme of the story automatically.
    e.g. jungle, space, temple, war, city, etc.
    """
    text = script_text.lower()
    if any(word in text for word in ["jungle", "forest", "animal", "sher", "bandar"]):
        return "jungle"
    elif any(word in text for word in ["space", "planet", "alien", "mars", "galaxy"]):
        return "space"
    elif any(word in text for word in ["temple", "god", "dev", "bhagwan", "shakti"]):
        return "temple"
    elif any(word in text for word in ["battle", "fight", "yudh", "war", "king"]):
        return "war"
    elif any(word in text for word in ["love", "romantic", "heart", "feeling", "prem"]):
        return "romantic"
    elif any(word in text for word in ["city", "street", "modern", "office", "work"]):
        return "city"
    else:
        return "generic"


def get_theme_background(theme: str) -> str:
    """
    Returns a suitable cinematic background video URL based on the detected theme.
    """
    backgrounds = {
        "jungle": [
            "https://cdn.pixabay.com/video/2023/03/15/15632-432987.mp4",
            "https://cdn.pixabay.com/video/2022/12/09/14422-412987.mp4"
        ],
        "space": [
            "https://cdn.pixabay.com/video/2023/02/14/15222-422987.mp4",
            "https://cdn.pixabay.com/video/2023/05/05/16542-435555.mp4"
        ],
        "temple": [
            "https://cdn.pixabay.com/video/2023/01/25/14922-422987.mp4",
            "https://cdn.pixabay.com/video/2023/07/11/17222-441444.mp4"
        ],
        "war": [
            "https://cdn.pixabay.com/video/2023/04/04/16122-433333.mp4",
            "https://cdn.pixabay.com/video/2022/11/20/14122-411111.mp4"
        ],
        "romantic": [
            "https://cdn.pixabay.com/video/2023/03/28/15742-432555.mp4",
            "https://cdn.pixabay.com/video/2023/06/08/17022-439222.mp4"
        ],
        "city": [
            "https://cdn.pixabay.com/video/2023/05/12/16632-437777.mp4",
            "https://cdn.pixabay.com/video/2022/10/19/13922-410444.mp4"
        ],
        "generic": [
            "https://cdn.pixabay.com/video/2023/07/20/17333-443333.mp4",
            "https://cdn.pixabay.com/video/2023/03/01/15522-431999.mp4"
        ]
    }
    return random.choice(backgrounds.get(theme, backgrounds["generic"]))


def add_theme_based_background(script_text: str, audio_path: str) -> str:
    """
    Combines detected theme background with the final mixed audio.
    """
    theme = detect_story_theme(script_text)
    bg_video = get_theme_background(theme)
    final_theme_video = f"/tmp/final_theme_scene_{uuid.uuid4().hex}.mp4"

    print(f"🌍 Theme detected: {theme} | 🎬 Using background: {bg_video}")

    # Merge audio and background
    os.system(f"ffmpeg -y -i {bg_video} -i {audio_path} -c:v libx264 -c:a aac -shortest {final_theme_video}")

    return final_theme_video

def build_auto_scene(script_text: str):
    """
    Builds cinematic video automatically by combining detected characters,
    their voices, and background visuals.
    """
    dialogues = generate_auto_scene_script(script_text)
    audio_segments = []

    try:
        for character, text in dialogues:
            print(f"🎙 Generating voice for {character}: '{text[:40]}...'")
            voice_id = CHARACTER_MODELS.get(character, {}).get("voice", "male")
            avatar = CHARACTER_MODELS.get(character, {}).get("avatar", "assets/avatars/default_loop.mp4")

            audio_path = f"/tmp/{character}_auto.mp3"
            audio = generate(text=text, voice=voice_id, model="eleven_turbo_v2")
            save(audio, audio_path)
            audio_segments.append(audio_path)

        # Merge all voices
        combined_audio = "/tmp/final_auto_mix.mp3"
        with open(combined_audio, "wb") as out_f:
            for path in audio_segments:
                with open(path, "rb") as in_f:
                    out_f.write(in_f.read())

        # Choose random cinematic background
        bg_list = [
            "https://cdn.pixabay.com/video/2023/07/18/17123-441987.mp4",
            "https://cdn.pixabay.com/video/2023/02/16/15423-422987.mp4",
            "https://cdn.pixabay.com/video/2023/05/25/16872-433222.mp4",
        ]
        bg_choice = random.choice(bg_list)

        final_video = f"/tmp/final_auto_scene_{uuid.uuid4().hex}.mp4"
        os.system(f"ffmpeg -y -i {bg_choice} -i {combined_audio} -c:v libx264 -c:a aac -shortest {final_video}")

        print(f"✅ Auto cinematic scene created: {final_video}")
        return final_video

    except Exception as e:
        print(f"❌ Auto scene build failed: {e}")
        return None

# fallback avatar (if not present)
FALLBACK_AVATAR = "assets/avatars/default_loop.mp4"

# ---------- Helper: character detection from script ----------
import re
def detect_char_dialogues(script_text: str) -> List[Tuple[str,str]]:
    """
    Parse script lines into list of (character, text).
    Accepts formats like:
      "Tiger: Hello" or "Man: Let's go" or free prose (then becomes narrator).
    """
    lines = [l.strip() for l in script_text.splitlines() if l.strip()]
    result = []
    for ln in lines:
        m = re.match(r"^([A-Za-z0-9_ ]+):\s*(.+)$", ln)
        if m:
            name = m.group(1).strip()
            text = m.group(2).strip()
        else:
            # fallback narrator
            name = "narrator"
            text = ln
        result.append((name, text))
    return result

def map_to_known_character(name: str) -> str:
    s = name.lower().strip()
    if any(x in s for x in ["man","boy","male","sir"]): return "man"
    if any(x in s for x in ["woman","female","girl","lady"]): return "woman"
    if any(x in s for x in ["child","kid","boy","girl"]): return "child"
    if any(x in s for x in ["old","grand","baba","dada","nana"]): return "old_man"
    if "god" in s or "lord" in s or "deva" in s: return "god"
    # animals explicit names
    for k in ("tiger","monkey","lion","elephant","fox"):
        if k in s: return k
    # robot/alien/narrator fallback
    if "robot" in s: return "robot"
    if "alien" in s: return "alien"
    if "narrator" in s: return "narrator"
    # default fallback
    return "man"

# ---------- Helper: generate voice for a character (ElevenLabs if available else gTTS) ----------
GTTS_AVAILABLE = 'gTTS' in globals()
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY") or os.getenv("ELEVENLABS_API_KEY") or os.getenv("ELEVEN_LABS_KEY")

def generate_character_voice_fallback(character_label: str, text: str) -> str:
    """
    Generates TTS audio file and returns local path.
    Prefer ElevenLabs via HTTP if ELEVEN_API_KEY present (simple POST),
    else use gTTS fallback.
    """
    uid = uuid.uuid4().hex
    out_mp3 = str(Path(app.config.get("TMP_FOLDER", "tmp"))/ f"{character_label}_{uid}.mp3")
    # Attempt ElevenLabs (simple HTTP) - if not configured, fallback to gTTS
    try:
        if ELEVEN_API_KEY:
            import requests
            voice_map = {
                "male":"21m00Tcm4TlvDq8ikWAM",
                "female":"EXAVITQu4vr4xnSDxMaL",
                "child":"TxGEqnHWrfWFTfGW9XjX",
                "old":"ErXwobaYiN019PkySvjV",
                "divine":"ErXwobaYiN019PkySvjV"
            }
            voice_id = voice_map.get(map_to_known_character(character_label), "21m00Tcm4TlvDq8ikWAM")
            payload = {"text": text}
            headers = {"xi-api-key": ELEVEN_API_KEY, "Content-Type":"application/json"}
            # ElevenLabs TTS endpoint template — may require adjustment per your account
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            r = requests.post(url, json=payload, headers=headers, stream=True, timeout=30)
            if r.status_code == 200:
                with open(out_mp3, "wb") as fh:
                    for chunk in r.iter_content(1024*32):
                        fh.write(chunk)
                return out_mp3
            # if fails, fallthrough to gTTS
    except Exception:
        pass

    # gTTS fallback
    try:
        from gtts import gTTS
        tts = gTTS(text, lang="hi" if re.search(r'[\u0900-\u097F]', text) else "en")
        Path(os.path.dirname(out_mp3)).mkdir(parents=True, exist_ok=True)
        tts.save(out_mp3)
        return out_mp3
    except Exception as e:
        log.exception("TTS fallback failed: %s", e)
        raise RuntimeError("TTS generation failed")

# ---------- Helper: create lip-synced clip by simply attaching audio to avatar loop ----------
def create_lip_sync_clip(avatar_path: str, audio_path: str, duration: float = None) -> str:
    """
    avatar_path: local path or url to an avatar loop video (prefer mp4)
    audio_path: local path to audio
    returns path to output mp4
    """
    tmpdir = Path(app.config.get("TMP_FOLDER","tmp"))
    tmpdir.mkdir(parents=True, exist_ok=True)
    out = tmpdir / f"lipsync_{uuid.uuid4().hex}.mp4"
    avatar = avatar_path or FALLBACK_AVATAR

    # if avatar is remote URL, download
    if str(avatar).startswith("http"):
        try:
            import requests
            resp = requests.get(avatar, stream=True, timeout=30)
            if resp.status_code == 200:
                avtmp = tmpdir / f"avatar_{uuid.uuid4().hex}.mp4"
                with open(avtmp, "wb") as f:
                    for chunk in resp.iter_content(8192):
                        f.write(chunk)
                avatar = str(avtmp)
        except Exception:
            avatar = FALLBACK_AVATAR

    # determine audio duration
    audio_dur = None
    try:
        from moviepy.editor import AudioFileClip
        ac = AudioFileClip(_abs_path(audio_path))
        audio_dur = ac.duration
        ac.close()
    except Exception:
        audio_dur = duration or 5.0

    # use ffmpeg to trim/or loop avatar to audio length, then attach audio
    # create video of exact audio_dur by looping avatar
    loop_video = tmpdir / f"avatar_loop_{uuid.uuid4().hex}.mp4"
    # -stream_loop for local files only; for remote already downloaded
    cmd_loop = f"ffmpeg -y -i {avatar} -filter_complex \"loop=loop=0:size=1:start=0\" -t {audio_dur} -c:v libx264 -c:a copy {loop_video}"
    # simpler fallback: just copy and let -shortest truncate
    cmd_concat = f"ffmpeg -y -i {avatar} -i {audio_path} -shortest -c:v libx264 -c:a aac {out}"
    try:
        # try straightforward attach (works even if durations mismatch)
        subprocess.check_call(cmd_concat, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return str(out)
    except subprocess.CalledProcessError:
        # final fallback: use moviepy concatenation
        try:
            from moviepy.editor import VideoFileClip, AudioFileClip
            v = VideoFileClip(_abs_path(avatar))
            a = AudioFileClip(_abs_path(audio_path))
            if a.duration and v.duration < a.duration:
                # loop video
                clips = []
                t = 0.0
                while t < a.duration:
                    clips.append(v.subclip(0, min(v.duration, a.duration - t)))
                    t += v.duration
                final = concatenate_videoclips(clips)
            else:
                final = v.subclip(0, a.duration) if a.duration else v
            final = final.set_audio(a)
            final.write_videofile(str(out), codec="libx264", audio_codec="aac")
            v.close(); a.close()
            return str(out)
        except Exception as e:
            log.exception("create_lip_sync_clip failed: %s", e)
            raise

# ---------- Compose scene: sequential clips (one clip per dialogue) ----------
def compose_visual_scene_from_dialogues(dialogues: List[Tuple[str,str]]) -> str:
    """
    For each (character, text) create voice + lip-synced avatar clip, then concat sequentially.
    Returns local mp4 path.
    """
    tmpdir = Path(app.config.get("TMP_FOLDER","tmp"))
    tmpdir.mkdir(parents=True, exist_ok=True)
    clip_paths = []
    for name, text in dialogues:
        char_key = map_to_known_character(name)
        model = CHARACTER_MODELS.get(char_key, {})
        avatar = model.get("avatar") or FALLBACK_AVATAR
        # generate voice
        try:
            audio_path = generate_character_voice_fallback(name, text)
        except Exception as e:
            log.exception("voice generation failed: %s", e)
            continue
        # create lipsync clip
        try:
            clip = create_lip_sync_clip(avatar, audio_path)
            clip_paths.append(clip)
        except Exception as e:
            log.exception("lip-sync clip failed for %s: %s", name, e)

    if not clip_paths:
        return None

    # concatenate clips into final scene
    final_out = tmpdir / f"ucve_scene_{uuid.uuid4().hex}.mp4"
    try:
        # build ffmpeg input string
        with open(tmpdir / "inputs.txt", "w") as fh:
            for p in clip_paths:
                fh.write(f"file '{p}'\n")
        cmd = f"ffmpeg -y -f concat -safe 0 -i {tmpdir/'inputs.txt'} -c copy {final_out}"
        subprocess.check_call(cmd, shell=True)
        return str(final_out)
    except Exception:
        # fallback to moviepy concatenation
        try:
            from moviepy.editor import VideoFileClip, concatenate_videoclips
            clips = [VideoFileClip(p) for p in clip_paths]
            final = concatenate_videoclips(clips, method="compose")
            outp = str(final_out)
            final.write_videofile(outp, codec="libx264", audio_codec="aac")
            for c in clips: c.close()
            return outp
        except Exception as e:
            log.exception("compose concat failed: %s", e)
            return None

# ---------- Public entry: Universal Character Visual Engine ----------
def generate_universal_scene(script_text: str) -> str:
    """
    Main entry: takes script_text, auto-detects characters, composes visual scene,
    then runs cinematic postprocessing (camera, ambient, lighting) if those functions exist.
    Returns path to final mp4 or None.
    """
    try:
        dialogues = detect_char_dialogues(script_text)
        # Map names to known characters keywords
        dialogues_mapped = [(map_to_known_character(n), t) for (n,t) in dialogues]
        scene = compose_visual_scene_from_dialogues(dialogues_mapped)
        if not scene:
            return None

        # optional postprocessing (use if functions defined earlier)
        try:
            # derive mood from full script if analyze_emotion exists
            mood = None
            if 'analyze_emotion' in globals():
                mood = analyze_emotion(script_text)
            else:
                mood = "neutral"

            if 'apply_dynamic_camera_effects' in globals():
                scene = apply_dynamic_camera_effects(scene, mood or "cinematic")
            if 'add_ambient_soundscape' in globals():
                # choose ambient based on mood
                amb = "jungle" if "tiger" in script_text.lower() or "jungle" in script_text.lower() else ("calm" if mood=="sad" else "emotional")
                scene = add_ambient_soundscape(scene, amb)
            if 'apply_cinematic_lighting' in globals():
                scene = apply_cinematic_lighting(scene, mood or "neutral")

        except Exception as e:
            log.exception("postprocessing failed: %s", e)

        return scene
    except Exception as e:
        log.exception("UCVE generation failed: %s", e)
        return None

# -------------------- Cinematic Background Music AI System --------------------
import random

# Predefined emotion-based music library (You can expand it)
CINEMATIC_MUSIC = {
    "happy": [
        "https://cdn.pixabay.com/audio/2023/03/28/audio_5a3c0b3d49.mp3",
        "https://cdn.pixabay.com/audio/2023/03/14/audio_5e7b85e59d.mp3"
    ],
    "sad": [
        "https://cdn.pixabay.com/audio/2023/02/07/audio_03d9d32b2a.mp3",
        "https://cdn.pixabay.com/audio/2022/12/27/audio_d55aaf7b24.mp3"
    ],
    "excited": [
        "https://cdn.pixabay.com/audio/2023/01/23/audio_6482b8f3b5.mp3",
        "https://cdn.pixabay.com/audio/2023/03/15/audio_80a07d6a37.mp3"
    ],
    "angry": [
        "https://cdn.pixabay.com/audio/2023/03/06/audio_14b8231a12.mp3",
        "https://cdn.pixabay.com/audio/2023/01/17/audio_b14c97f203.mp3"
    ],
    "romantic": [
        "https://cdn.pixabay.com/audio/2022/11/15/audio_d8579a3b47.mp3",
        "https://cdn.pixabay.com/audio/2023/02/10/audio_2b47d16e21.mp3"
    ],
    "neutral": [
        "https://cdn.pixabay.com/audio/2023/02/28/audio_04e4b09c56.mp3"
    ]
}

def select_music_for_emotion(emotion: str) -> str:
    """
    Pick a random cinematic track for a given emotion.
    """
    try:
        tracks = CINEMATIC_MUSIC.get(emotion, CINEMATIC_MUSIC["neutral"])
        return random.choice(tracks)
    except Exception as e:
        log.warning(f"Music select failed for {emotion}: {e}")
        return CINEMATIC_MUSIC["neutral"][0]


def build_full_cinematic_story(script_text: str):
    """
    Combine emotion, avatar, voice, and background music into a cinematic movie.
    """
    segments = detect_characters_from_script(script_text)
    clips = []

    for seg in segments:
        name = seg["character"]
        voice_type = seg["voice_type"]
        text = seg["text"]
        emotion = analyze_emotion_from_text(text)
        avatar_url = fetch_avatar_for_character(name, voice_type)
        bg_music = select_music_for_emotion(emotion)

        log.info(f"🎬 Scene: {name} ({emotion}) with music → {bg_music}")

        try:
            img_clip = ImageClip(avatar_url).resize((720, 1280))
            voice_path = generate_character_voice(name, text)
            audio_clip = AudioFileClip(voice_path)
            bg_clip = AudioFileClip(bg_music).volumex(0.1)

            dur = max(audio_clip.duration, 3)
            img_clip = cinematic_motion(img_clip)
            img_clip = apply_emotion_filter(img_clip, emotion)

            # merge audio layers
            final_audio = CompositeAudioClip([audio_clip, bg_clip])
            final_clip = img_clip.set_duration(dur).set_audio(final_audio)
            clips.append(final_clip)
        except Exception as e:
            log.warning(f"Scene music merge failed: {e}")

    if clips:
        final_video = concatenate_videoclips(clips, method="compose")
        output_path = OUTPUT_FOLDER / f"cinematic_story_{uuid.uuid4().hex}.mp4"
        final_video.write_videofile(str(output_path), fps=24)
        log.info(f"🎞 Full cinematic story with background music: {output_path}")
        return str(output_path)
    else:
        log.warning("⚠️ No scenes generated.")
        return None

# -------------------- AI Scene Composer System (Emotion-based Background Generator) --------------------
import io
from PIL import Image

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

def generate_scene_background(text: str, emotion: str) -> str:
    """
    Generate AI background image using emotion + text prompt.
    Uses OpenAI Image API (DALL·E style fallback).
    """
    try:
        if not OPENAI_API_KEY:
            log.warning("⚠️ No OpenAI API key found, using fallback background.")
            return random.choice([
                "https://images.unsplash.com/photo-1529626455594-4ff0802cfb7e",
                "https://images.unsplash.com/photo-1506744038136-46273834b3fb",
                "https://images.unsplash.com/photo-1501785888041-af3ef285b470"
            ])

        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        prompt = f"{emotion} cinematic background, 4K ultra realistic, movie lighting, {text}"
        data = {"model": "gpt-image-1", "prompt": prompt, "size": "1024x1024"}
        r = requests.post("https://api.openai.com/v1/images/generations", headers=headers, json=data)

        if r.status_code == 200:
            image_url = r.json()["data"][0]["url"]
            log.info(f"🖼 AI background generated for {emotion}: {image_url}")
            return image_url
        else:
            log.warning(f"AI background gen failed: {r.text}")
            return random.choice([
                "https://images.unsplash.com/photo-1519125323398-675f0ddb6308",
                "https://images.unsplash.com/photo-1506744038136-46273834b3fb"
            ])
    except Exception as e:
        log.warning(f"AI Scene Composer Error: {e}")
        return "https://images.unsplash.com/photo-1529626455594-4ff0802cfb7e"


def build_ai_composed_video(script_text: str):
    """
    Combine AI-generated backgrounds + voices + avatars + music → cinematic movie.
    """
    segments = detect_characters_from_script(script_text)
    clips = []

    for seg in segments:
        name = seg["character"]
        voice_type = seg["voice_type"]
        text = seg["text"]
        emotion = analyze_emotion_from_text(text)

        bg_image = generate_scene_background(text, emotion)
        avatar_url = fetch_avatar_for_character(name, voice_type)
        voice_path = generate_character_voice(name, text)
        bg_music = select_music_for_emotion(emotion)

        log.info(f"🎬 Scene composed: {name} ({emotion}) | BG: {bg_image}")

        try:
            # Create layered composition: background + avatar + motion
            bg_clip = ImageClip(bg_image).resize((720, 1280))
            avatar_clip = ImageClip(avatar_url).resize((400, 400)).set_position(("center", "center"))
            voice_clip = AudioFileClip(voice_path)
            music_clip = AudioFileClip(bg_music).volumex(0.08)

            dur = max(voice_clip.duration, 4)
            bg_clip = apply_emotion_filter(bg_clip, emotion)
            avatar_clip = cinematic_motion(avatar_clip)

            scene = CompositeVideoClip([bg_clip, avatar_clip]).set_duration(dur)
            final_audio = CompositeAudioClip([voice_clip, music_clip])
            final_scene = scene.set_audio(final_audio)

            clips.append(final_scene)
        except Exception as e:
            log.warning(f"Scene composition failed for {name}: {e}")

    if clips:
        final_video = concatenate_videoclips(clips, method="compose")
        output_path = OUTPUT_FOLDER / f"ai_scene_story_{uuid.uuid4().hex}.mp4"
        final_video.write_videofile(str(output_path), fps=24)
        log.info(f"🎞 AI Scene Composed Cinematic Video: {output_path}")
        return str(output_path)
    else:
        log.warning("⚠️ No scenes generated for AI composition.")
        return None

# -------------------- AI Lip-Sync System (D-ID integration + MoviePy fallback) --------------------
import mimetypes

DID_API_KEY = os.getenv("DID_API_KEY", "")
DID_API_URL = os.getenv("DID_API_URL", "https://api.d-id.com")

def call_did_talks(image_path: str, audio_path: Optional[str], script_text: Optional[str] = None):
    """
    Call D-ID 'talks' (or equivalent) endpoint to create a talking-head video.
    Returns path to the saved mp4 or None on failure.

    NOTE: D-ID API details may change — this is a template. Replace payload/endpoint
    according to the D-ID docs if required.
    """
    if not DID_API_KEY:
        return None

    try:
        headers = {"Authorization": f"Bearer {DID_API_KEY}"}
        url = f"{DID_API_URL}/talks"

        # Prepare files (image must be uploaded as multipart)
        files = {}
        data = {}

        # image
        mime, _ = mimetypes.guess_type(image_path)
        files["image"] = ("avatar", open(image_path, "rb"), mime or "image/png")

        # audio: if audio_path provided, attach; else we send script (text) for their TTS
        if audio_path:
            files["audio"] = ("audio", open(audio_path, "rb"), "audio/mpeg")
        elif script_text:
            data["script"] = script_text

        # optional params: voice, crop, background etc. depends on D-ID API
        data["driver"] = "d-id"  # placeholder
        # send request
        r = requests.post(url, headers=headers, data=data, files=files, timeout=120)

        # close opened files
        try:
            files["image"][1].close()
        except: pass
        if "audio" in files:
            try: files["audio"][1].close()
            except: pass

        if r.status_code in (200,201):
            j = r.json()
            # D-ID typically returns job id / video url — template below:
            video_url = j.get("result_url") or j.get("video_url") or j.get("output", {}).get("video_url")
            if video_url:
                # download video to outputs
                out = OUTPUT_DIR / f"lipsync_{uuid.uuid4().hex}.mp4"
                resp = requests.get(video_url, stream=True, timeout=120)
                if resp.status_code == 200:
                    with open(out, "wb") as fh:
                        for chunk in resp.iter_content(1024 * 32):
                            fh.write(chunk)
                    return str(out)
        log.warning("D-ID call failed: %s %s", r.status_code, r.text)
    except Exception as e:
        log.exception("D-ID integration error: %s", e)
    return None

# ==============================================
# 🎭 EMOTION-BASED VIDEO ENHANCER MODULE
# ==============================================
from textblob import TextBlob

def analyze_emotion(text):
    sentiment = TextBlob(text).sentiment.polarity
    if sentiment > 0.3:
        return "happy"
    elif sentiment < -0.3:
        return "sad"
    elif -0.3 <= sentiment <= 0.3:
        return "neutral"
    else:
        return "angry"

def apply_emotion_effects(emotion, input_video, output_video):
    try:
        if emotion == "happy":
            os.system(f"ffmpeg -y -i {input_video} -vf 'zoompan=z=\'min(zoom+0.0015,1.3)\':d=1:s=1080x1920' -af 'volume=1.2' {output_video}")
        elif emotion == "sad":
            os.system(f"ffmpeg -y -i {input_video} -vf 'hue=s=0.6, eq=brightness=-0.05' -af 'volume=0.8' {output_video}")
        elif emotion == "angry":
            os.system(f"ffmpeg -y -i {input_video} -vf 'eq=contrast=1.5:saturation=1.4' -af 'volume=1.5' {output_video}")
        else:
            os.system(f"cp {input_video} {output_video}")
        return output_video
    except Exception as e:
        print("❌ Emotion effect error:", e)
        return input_video

# --------- Fallback (MoviePy based approximate lip-sync) ----------
def approximate_lipsync_moviepy(image_path: str, audio_path: str, out_path: str, mouth_overlay_path: Optional[str] = None):
    """
    Create an approximate lip-sync video by toggling a 'mouth open' overlay or
    by applying small transforms synced to audio energy.
    - image_path: base face image (local path or URL)
    - audio_path: local audio path (mp3/wav)
    - mouth_overlay_path: optional PNG with transparent mouth-open region to overlay
    """
    if not MOVIEPY_AVAILABLE:
        raise RuntimeError("MoviePy not available for fallback lipsync")

    # resolve image local path or download if URL
    img_local = image_path
    if str(image_path).startswith("http"):
        tmp = UPLOAD_DIR / f"lipsync_img_{uuid.uuid4().hex}.png"
        try:
            r = requests.get(image_path, stream=True, timeout=30)
            if r.status_code == 200:
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(8192):
                        fh.write(chunk)
                img_local = str(tmp)
        except Exception as e:
            log.warning("Failed to download avatar image for lipsync: %s", e)
            img_local = image_path  # let MoviePy try

    audio_clip = AudioFileClip(_abs_path(audio_path))
    duration = audio_clip.duration
    base = ImageClip(_abs_path(img_local)).set_duration(duration).resize(width=720)

    # sample short windows of audio and compute RMS -> use as open/close threshold
    try:
        arr = audio_clip.to_soundarray(nbytes=2, fps=22050)
        # arr shape (N, channels); compute mono RMS per sample chunk
        import numpy as np
        mono = arr.mean(axis=1) if arr.ndim == 2 else arr
        chunk = 22050 // 10  # 0.1s chunks
        rms_vals = []
        for i in range(0, len(mono), chunk):
            seg = mono[i:i+chunk]
            rms = (seg.astype(float) ** 2).mean() ** 0.5 if len(seg) > 0 else 0.0
            rms_vals.append(rms)
        # normalize
        maxr = max(rms_vals) if rms_vals else 1.0
        rms_norm = [v / maxr for v in rms_vals]
    except Exception:
        # fallback simple periodic toggle if sound analysis fails
        rms_norm = [0.2 if (i % 2 == 0) else 0.8 for i in range(int(duration * 10))]

    # build small subclips and overlay mouth_open when rms above threshold
    subclips = []
    t = 0.0
    idx = 0
    seg_dur = 0.1
    while t < duration - 1e-6:
        segt = min(seg_dur, duration - t)
        rmsv = rms_norm[min(idx, len(rms_norm)-1)]
        clip = base.subclip(t, t+segt)
        # apply tiny scale/pulse to simulate speaking
        if rmsv > 0.25:
            clip = clip.fx(resize, 1.01).fx(lambda c: c)  # small zoom when speaking
            # overlay mouth image if provided
            if mouth_overlay_path and Path(mouth_overlay_path).exists():
                mouth = (ImageClip(str(mouth_overlay_path))
                         .set_duration(segt)
                         .resize(width=int(base.w * 0.35))
                         .set_position(("center", int(base.h*0.6))))
                comp = CompositeVideoClip([clip, mouth])
                subclips.append(comp)
            else:
                subclips.append(clip)
        else:
            # quiet frame
            subclips.append(clip)
        t += segt
        idx += 1

    final = concatenate_videoclips(subclips, method="compose")
    final = final.set_audio(audio_clip)
    # optional add bg music or filters here
    bitrate = "2500k"
    final.write_videofile(out_path, fps=24, codec="libx264", audio_codec="aac", bitrate=bitrate)
    final.close()
    audio_clip.close()
    return out_path


# ------------ Endpoint: /generate_lipsync ---------------
@app.route("/generate_lipsync", methods=["POST"])
def api_generate_lipsync():
    """
    Accepts multipart/form-data:
      - image (character image URL or uploaded file)
      - audio (uploaded audio file OR script+voice will be used)
      - or script + character name (then backend will generate voice via generate_character_voice)
      - use_did = true/false (prefer D-ID if available)
    Returns: job_id and will create output under outputs/
    """
    user = request.form.get("user_email", "demo@visora.com")
    use_did = request.form.get("use_did", "true").lower() == "true"

    # prefer uploaded image
    image_local = None
    if "image" in request.files:
        f = request.files["image"]
        fname = secure_filename(f.filename)
        dest = UPLOAD_DIR / "lipsync_images" / f"{uuid.uuid4().hex}_{fname}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        f.save(dest)
        image_local = str(dest)
    else:
        image_url = request.form.get("image_url")
        image_local = image_url

    audio_local = None
    if "audio" in request.files:
        f = request.files["audio"]
        fname = secure_filename(f.filename)
        dest = UPLOAD_DIR / "lipsync_audio" / f"{uuid.uuid4().hex}_{fname}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        f.save(dest)
        audio_local = str(dest)
    else:
        # script mode: generate TTS
        script = request.form.get("script","")
        character = request.form.get("character","C1")
        if not script:
            return jsonify({"error":"no audio or script provided"}), 400
        audio_local = generate_character_voice(character, script)
        if not audio_local:
            return jsonify({"error":"tts failed"}), 500

    # create DB record
    v = Video(user_email=user, title=f"LipSync_{uuid.uuid4().hex[:6]}", status="queued", meta=json.dumps({"type":"lipsync"}))
    db.session.add(v); db.session.commit()

    job_id = uuid.uuid4().hex
    render_jobs[job_id] = {"job_id": job_id, "status": "queued", "created_at": datetime.utcnow().isoformat()}
    # try D-ID if requested
    if use_did and DID_API_KEY:
        try:
            out = call_did_talks(image_local, audio_local, script_text=request.form.get("script"))
            if out:
                render_jobs[job_id]["status"] = "done"
                render_jobs[job_id]["output"] = str(Path(out).relative_to(BASE_DIR))
                # update DB
                v.file_path = render_jobs[job_id]["output"]; v.status="done"; db.session.commit()
                return jsonify({"job_id": job_id, "output": render_jobs[job_id]["output"]})
        except Exception as e:
            log.warning("D-ID attempt failed, falling back to local lip-sync: %s", e)

    # fallback: local approximate lipsync
    try:
        out_abs = str(OUTPUT_DIR / f"lipsync_{uuid.uuid4().hex}.mp4")
        approximate_lipsync_moviepy(image_local, audio_local, out_abs)
        render_jobs[job_id]["status"] = "done"
        render_jobs[job_id]["output"] = str(Path(out_abs).relative_to(BASE_DIR))
        v.file_path = render_jobs[job_id]["output"]; v.status="done"; db.session.commit()
        return jsonify({"job_id": job_id, "output": render_jobs[job_id]["output"]})
    except Exception as e:
        log.exception("Local lipsync failed: %s", e)
        render_jobs[job_id]["status"] = "failed"; render_jobs[job_id]["error"] = str(e)
        v.status = "failed"; db.session.commit()
        return jsonify({"error":"lipsync_failed","details":str(e)}), 500

# ==============================================
# 🎭 EMOTION-BASED VIDEO ENHANCER MODULE
# ==============================================
from textblob import TextBlob

def analyze_emotion(text):
    sentiment = TextBlob(text).sentiment.polarity
    if sentiment > 0.3:
        return "happy"
    elif sentiment < -0.3:
        return "sad"
    elif -0.3 <= sentiment <= 0.3:
        return "neutral"
    else:
        return "angry"

def apply_emotion_effects(emotion, input_video, output_video):
    try:
        if emotion == "happy":
            os.system(f"ffmpeg -y -i {input_video} -vf 'zoompan=z=\'min(zoom+0.0015,1.3)\':d=1:s=1080x1920' -af 'volume=1.2' {output_video}")
        elif emotion == "sad":
            os.system(f"ffmpeg -y -i {input_video} -vf 'hue=s=0.6, eq=brightness=-0.05' -af 'volume=0.8' {output_video}")
        elif emotion == "angry":
            os.system(f"ffmpeg -y -i {input_video} -vf 'eq=contrast=1.5:saturation=1.4' -af 'volume=1.5' {output_video}")
        else:
            os.system(f"cp {input_video} {output_video}")
        return output_video
    except Exception as e:
        print("❌ Emotion effect error:", e)
        return input_video

# -------------------- Google Veo Cinematic Generator --------------------
import base64

GOOGLE_VEO_API_KEY = os.getenv("GOOGLE_VEO_API_KEY", "")

def generate_cinematic_video(prompt: str, duration: int = 10):
    """
    Generate cinematic short video using Google Veo (if available)
    Fallback: moviepy placeholder (color + text)
    """
    out_dir = OUTPUT_FOLDER
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"veo_{uuid.uuid4().hex}.mp4"

    if GOOGLE_VEO_API_KEY:
        try:
            url = "https://generativelanguage.googleapis.com/v1beta/models/veo:generateVideo"
            params = {"key": GOOGLE_VEO_API_KEY}
            headers = {"Content-Type": "application/json"}
            payload = {
                "prompt": {"text": prompt},
                "videoConfig": {"durationSeconds": duration, "aspectRatio": "16:9"},
            }
            r = requests.post(url, headers=headers, json=payload, params=params, timeout=120)
            if r.status_code == 200:
                data = r.json()
                if "video" in data and "base64Data" in data["video"]:
                    video_data = base64.b64decode(data["video"]["base64Data"])
                    with open(out_path, "wb") as f:
                        f.write(video_data)
                    log.info(f"Google Veo cinematic video created: {out_path}")
                    return str(out_path)
                else:
                    log.warning("Veo API response incomplete, fallbacking.")
            else:
                log.warning("Veo API error: %s", r.text)
        except Exception as e:
            log.exception("Google Veo generation failed: %s", e)

    # fallback MoviePy cinematic placeholder
    try:
        from moviepy.editor import TextClip, ColorClip, CompositeVideoClip
        text = TextClip(prompt, fontsize=50, color='white', bg_color=None, method='caption', size=(1280,720))
        bg = ColorClip(size=(1280,720), color=(10,10,10)).set_duration(duration)
        final = CompositeVideoClip([bg, text.set_position("center").set_duration(duration)])
        final.write_videofile(str(out_path), fps=24, codec="libx264", audio=False)
        final.close()
        log.info(f"Fallback cinematic video created: {out_path}")
        return str(out_path)
    except Exception as e:
        log.exception("Fallback cinematic video failed: %s", e)
        return None


@app.route("/generate_cinematic", methods=["POST"])
def generate_cinematic():
    """
    JSON Example:
      {
        "user_email": "demo@visora.com",
        "prompt": "A cinematic sunrise over the mountains with birds flying",
        "duration": 10
      }
    """
    data = request.get_json(force=True)
    user_email = data.get("user_email", "demo@visora.com")
    prompt = data.get("prompt", "")
    duration = int(data.get("duration", 10))

    if not prompt.strip():
        return jsonify({"error": "prompt required"}), 400

    output_path = generate_cinematic_video(prompt, duration)
    if not output_path:
        return jsonify({"error": "generation failed"}), 500

    v = UserVideo(user_email=user_email, title=f"Cinematic_{uuid.uuid4().hex[:6]}",
                  file_path=str(Path(output_path).relative_to(BASE_DIR)),
                  status="done", meta_json=json.dumps({"prompt": prompt, "mode": "cinematic"}))
    db.session.add(v)
    db.session.commit()
    return jsonify({"message": "cinematic video created", "file": v.file_path, "video_id": v.id}

# =====================================================
# 🎥 MULTI-SCENE CINEMATIC BACKGROUND TRANSITION ENGINE
# =====================================================
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip, CompositeVideoClip, vfx

def build_multiscene_video(dialogues):
    """
    Builds multi-scene cinematic video with background transitions.
    Each dialogue = 1 scene with fade, zoom, and unique background.
    """
    try:
        print("🎬 Building multi-scene cinematic video...")
        scene_clips = []
        backgrounds = [
            "https://cdn.pixabay.com/video/2023/03/10/158989-804662703_large.mp4",  # Forest
            "https://cdn.pixabay.com/video/2023/06/12/168662-836232104_large.mp4",  # Sunset
            "https://cdn.pixabay.com/video/2021/09/08/90551-590322904_large.mp4",  # Mountain
            "https://cdn.pixabay.com/video/2024/02/02/201425-902930285_large.mp4"   # River
        ]

        # Create a short scene for each dialogue
        for idx, (animal, text) in enumerate(dialogues):
            bg_path = random.choice(backgrounds)
            audio_path = f"/tmp/{animal}_voice.mp3"
            video_clip = VideoFileClip(bg_path).subclip(0, 5).resize((1080, 1920))
            audio_clip = AudioFileClip(audio_path)

            # Apply cinematic zoom and fade transitions
            zoomed = video_clip.fx(vfx.zoom_in, 1.1 + 0.05 * idx)
            faded = zoomed.crossfadein(1).crossfadeout(1)

            # Add voice to background
            combined = faded.set_audio(audio_clip)
            scene_clips.append(combined)

        # Concatenate all scenes together
        final_movie = concatenate_videoclips(scene_clips, method="compose")
        output_path = f"/tmp/final_cinematic_story_{uuid.uuid4().hex}.mp4"
        final_movie.write_videofile(output_path, codec="libx264", audio_codec="aac")

        print(f"✅ Multi-scene cinematic video created: {output_path}")
        return output_path

    except Exception as e:
        print(f"❌ Multi-scene generation failed: {e}")
        return None

# =====================================================
# 🐯🦊🐒 MULTI-CHARACTER ANIMAL CINEMATIC VOICE ENGINE
# =====================================================
from random import choice

# Character voice map (ElevenLabs IDs or local tts)
ANIMAL_VOICES = {
    "tiger": "21m00Tcm4TlvDq8ikWAM",
    "monkey": "EXAVITQu4vr4xnSDxMaL",
    "lion": "TxGEqnHWrfWFTfGW9XjX",
    "fox": "ErXwobaYiN019PkySvjV",
    "elephant": "VR6AewLTigWG4xSOukaG",
    "eagle": "Xb7hH8MSUJpSbSDYk0k2"
}

def generate_animal_voice_script(script_text):
    """
    Splits text by dialogues and assigns animal voices automatically
    Example Input:
      "Tiger: Main sher hoon! Monkey: Mujhe to mazak chahiye!"
    """
    lines = script_text.split(" ")
    dialogues = []
    current_animal = None
    current_text = []

    for word in lines:
        if word.strip(":").lower() in ANIMAL_VOICES.keys():
            if current_animal and current_text:
                dialogues.append((current_animal, " ".join(current_text)))
            current_animal = word.strip(":").lower()
            current_text = []
        else:
            current_text.append(word)
    if current_animal and current_text:
        dialogues.append((current_animal, " ".join(current_text)))
    return dialogues

try:
    print("🎞 Building multi-character animal cinematic video...")
    from textblob import TextBlob

    audio_segments = []
    temp_audios = []

    # 🎤 Generate each animal's voice
    for animal, text in dialogues:
        print(f"🎙 Generating voice for {animal} → {text[:40]}...")
        voice_id = ANIMAL_VOICES.get(animal, ANIMAL_VOICES["lion"])
        audio_path = f"/tmp/{animal}_voice.mp3"
        audio = generate(text=text, voice=voice_id, model="eleven_multilingual_v2")
        save(audio, audio_path)
        temp_audios.append(audio_path)

    # 🎧 Merge all voices sequentially
    combined_audio = "/tmp/final_animals_mix.mp3"
    with open(combined_audio, "wb") as out_f:
        for path in temp_audios:
            with open(path, "rb") as in_f:
                out_f.write(in_f.read())

    # 🌲 Choose cinematic background
    bg_options = [
        "https://cdn.pixabay.com/video/2023/03/10/158989-804662703_large.mp4",
        "https://cdn.pixabay.com/video/2023/06/12/168662-836232104_large.mp4",
        "https://cdn.pixabay.com/video/2021/09/08/90551-590322904_large.mp4"
    ]
    bg_choice = random.choice(bg_options)

    # 🎨 AI Scene Mood & Color Grading
    mood = "neutral"
    try:
        all_text = " ".join([t for _, t in dialogues])
        sentiment = TextBlob(all_text).sentiment.polarity
        if sentiment > 0.4:
            mood = "warm"
        elif sentiment < -0.4:
            mood = "cool"
        else:
            mood = "cinematic"
    except Exception as e:
        print(f"⚠️ Sentiment check failed: {e}")

    # 🪄 Apply cinematic filters based on mood
    filter_cmd = ""
    if mood == "warm":
        filter_cmd = "-vf eq=brightness=0.05:saturation=1.3:contrast=1.2"
    elif mood == "cool":
        filter_cmd = "-vf eq=brightness=-0.03:saturation=0.8:contrast=1.0"
    elif mood == "cinematic":
        filter_cmd = "-vf curves=vintage"

    final_video = f"/tmp/animal_scene_{uuid.uuid4().hex}.mp4"
    os.system(f"ffmpeg -y -i {bg_choice} -i {combined_audio} -shortest {filter_cmd} -c:v libx264 -c:a aac {final_video}")

    print(f"✅ Animal cinematic video created successfully: {final_video}")
    return final_video

# 💡 AI Cinematic Lighting Engine
def apply_cinematic_lighting(input_video, mood="neutral"):
    """
    Dynamically adjusts brightness, contrast, and color tone
    based on mood to achieve a cinematic visual feel.
    Mood options: happy, sad, angry, emotional, dark, neutral
    """
    output_path = f"/tmp/lighting_{uuid.uuid4().hex}.mp4"

    try:
        print(f"💡 Applying cinematic lighting tone for mood: {mood}")

        lighting_map = {
            "happy": "eq=brightness=0.1:contrast=1.2:saturation=1.4",
            "sad": "eq=brightness=-0.1:contrast=0.9:saturation=0.7",
            "angry": "eq=contrast=1.5:saturation=1.1",
            "emotional": "eq=brightness=-0.05:contrast=1.1:saturation=1.0",
            "dark": "eq=brightness=-0.2:contrast=1.4:saturation=0.8",
            "neutral": "eq=brightness=0:contrast=1:saturation=1"
        }

        filter_value = lighting_map.get(mood, lighting_map["neutral"])

        os.system(f"ffmpeg -y -i {input_video} -vf {filter_value} -c:a copy {output_path}")

        print(f"✅ Cinematic lighting applied: {output_path}")
        return output_path

    except Exception as e:
        print(f"⚠️ Lighting engine error: {e}")
        return input_video

# 🎧 AI Ambient Soundscape Engine
def add_ambient_soundscape(input_video, mood="forest"):
    """
    Adds ambient cinematic sound effects based on scene mood.
    Mood options: forest, jungle, storm, calm, emotional
    """
    output_path = f"/tmp/ambient_mix_{uuid.uuid4().hex}.mp4"

    # Default ambient library
    ambient_map = {
        "forest": "https://cdn.pixabay.com/audio/2022/03/15/audio_2e69a0f9d6.mp3",
        "jungle": "https://cdn.pixabay.com/audio/2023/01/07/audio_65b8d8f239.mp3",
        "storm": "https://cdn.pixabay.com/audio/2022/08/25/audio_09b4e5e835.mp3",
        "calm": "https://cdn.pixabay.com/audio/2022/11/09/audio_941d8a07e1.mp3",
        "emotional": "https://cdn.pixabay.com/audio/2022/02/17/audio_c17a3b76c2.mp3",
    }

    ambient_sound = ambient_map.get(mood, ambient_map["forest"])

    try:
        print(f"🎶 Adding ambient soundscape: {mood}")

        os.system(
            f"ffmpeg -y -i {input_video} -i {ambient_sound} "
            f"-filter_complex \"[1:a]volume=0.5[a1];[0:a][a1]amix=inputs=2:duration=first[aout]\" "
            f"-map 0:v -map \"[aout]\" -shortest -c:v copy -c:a aac {output_path}"
        )

        print(f"✅ Ambient sound added → {output_path}")
        return output_path

    except Exception as e:
        print(f"⚠️ Ambient mix error: {e}")
        return input_video

# 🎥 AI Dynamic Camera Movement Engine
def apply_dynamic_camera_effects(input_video, mood):
    """
    Adds smooth cinematic camera motion (zoom, pan, shake, focus) based on mood.
    """
    output_path = f"/tmp/camera_effect_{uuid.uuid4().hex}.mp4"

    try:
        print(f"🎬 Applying AI camera effects for mood: {mood}")

        if mood == "warm":
            # Slow zoom-in for motivational or happy tone
            effect_cmd = (
                "-filter_complex "
                "\"zoompan=z='min(zoom+0.0015,1.2)':d=125:s=1280x720,eq=brightness=0.03:saturation=1.2\""
            )
        elif mood == "cool":
            # Slow zoom-out for calm or sad tone
            effect_cmd = (
                "-filter_complex "
                "\"zoompan=z='if(lte(zoom,1.0),1.2,max(zoom-0.0015,1.0))':d=125:s=1280x720,eq=saturation=0.8\""
            )
        elif mood == "cinematic":
            # Gentle horizontal pan (left-right) + slight vignette
            effect_cmd = (
                "-filter_complex "
                "\"crop=in_w-20:in_h-20,eq=contrast=1.1:saturation=1.05,vignette=PI/3\""
            )
        else:
            # Subtle shake for strong or dramatic emotion
            effect_cmd = (
                "-filter_complex "
                "\"vibrance=intensity=0.3,noise=alls=20:allf=t+u\""
            )

        os.system(f"ffmpeg -y -i {input_video} {effect_cmd} -c:v libx264 -preset veryfast -c:a copy {output_path}")
        print(f"🎥 Camera motion added successfully → {output_path}")
        return output_path

    except Exception as e:
        print(f"⚠️ Camera effect error: {e}")
        return input_video

except Exception as e:
    print(f"❌ Error building animal cinematic video: {e}")
    return None

def build_animal_scene(dialogues):
    """
    Builds cinematic video with multiple animal dialogues and voice-over.
    """
    audio_segments = []
    for animal, text in dialogues:
        try:
            print(f"🎙️ Generating voice for {animal}...")
            voice_id = ANIMAL_VOICES.get(animal, ANIMAL_VOICES["lion"])
            audio_path = f"/tmp/{animal}_voice.mp3"
            audio = generate(text=text, voice=voice_id, model="eleven_multilingual_v2")
            save(audio, audio_path)
            audio_segments.append(audio_path)
        except Exception as e:
            print(f"⚠️ Voice generation failed for {animal}: {e}")

    # merge audio clips into one file
    combined_audio = "/tmp/final_animals_mix.mp3"
    with open(combined_audio, "wb") as out_f:
        for path in audio_segments:
            with open(path, "rb") as in_f:
                out_f.write(in_f.read())

# Background cinematic visuals
# 🎥 Add dynamic camera motion
final_video = apply_dynamic_camera_effects(final_video, "cinematic")

# 🎧 Add ambient soundscape (forest, jungle, storm, calm, etc.)
final_video = add_ambient_soundscape(final_video, "jungle")

# 💡 Add cinematic lighting tone
final_video = apply_cinematic_lighting(final_video, "emotional")

# 🌈 Add theme-based background (auto jungle/space/temple/war/city)
final_video = add_theme_based_background(script_text, combined_audio)

return final_video

# ============================================
# 🎙️ Hybrid Voice + AI Character Generator
# ============================================

from pydub import AudioSegment
import tempfile
import os

@app.route("/generate-hybrid-video", methods=["POST"])
def generate_hybrid_video():
    """
    User apni khud ki voice upload kare aur system baki characters aur visuals auto add kare.
    """
    try:
        script_text = request.form.get("script_text", "")
        user_audio = request.files.get("user_audio")

        if not script_text.strip():
            return jsonify({"error": "Script text is required"}), 400

        if not user_audio:
            return jsonify({"error": "User audio file is missing"}), 400

        # Save user audio temporarily
        user_audio_path = f"/tmp/user_voice_{uuid.uuid4().hex}.wav"
        user_audio.save(user_audio_path)

        print(f"🎤 User audio received: {user_audio_path}")

        # Clean the user audio (normalize, remove noise, etc.)
        clean_audio_path = f"/tmp/clean_voice_{uuid.uuid4().hex}.wav"
        sound = AudioSegment.from_file(user_audio_path)
        sound = sound.normalize()
        sound.export(clean_audio_path, format="wav")
        print("🧼 Voice cleaned and normalized")

        # Detect missing characters and generate AI voices for them
        dialogues = generate_animal_voice_script(script_text)
        generated_segments = []

        for animal, text in dialogues:
            if "user" in animal.lower():
                generated_segments.append(clean_audio_path)
            else:
                try:
                    voice_id = ANIMAL_VOICES.get(animal, ANIMAL_VOICES["tiger"])
                    audio_path = f"/tmp/{animal}_auto_voice.mp3"
                    audio = generate(text=text, voice=voice_id, model="eleven_multilingual_v2")
                    save(audio, audio_path)
                    generated_segments.append(audio_path)
                    print(f"✅ Generated AI voice for {animal}")
                except Exception as e:
                    print(f"⚠️ Voice generation failed for {animal}: {e}")

        # Merge all voices
        combined_audio = f"/tmp/final_hybrid_audio_{uuid.uuid4().hex}.mp3"
        with open(combined_audio, "wb") as out_f:
            for path in generated_segments:
                with open(path, "rb") as in_f:
                    out_f.write(in_f.read())

        # Detect theme + visuals
        final_video = add_theme_based_background(script_text, combined_audio)

        print(f"🎬 Final hybrid video created: {final_video}")
        return jsonify({
            "message": "Hybrid video created successfully!",
            "video_path": final_video
        }), 200

    except Exception as e:
        print(f"❌ Hybrid generation failed: {e}")
        return jsonify({"error": str(e)}), 500

# ============================================
# 🎭 AI Face Avatar Generator (User Image + Voice)
# ============================================

from PIL import Image
import base64
import cv2
import numpy as np

@app.route("/generate-avatar-video", methods=["POST"])
def generate_avatar_video():
    """
    User apne photo aur voice (optional) se animated avatar video banata hai.
    Agar user voice nahi deta to AI voice use hoti hai.
    """
    try:
        script_text = request.form.get("script_text", "")
        user_photo = request.files.get("user_photo")
        user_audio = request.files.get("user_audio")

        if not user_photo:
            return jsonify({"error": "User photo missing"}), 400

        if not script_text.strip():
            return jsonify({"error": "Script text required"}), 400

        # Save user photo
        user_photo_path = f"/tmp/user_photo_{uuid.uuid4().hex}.jpg"
        user_photo.save(user_photo_path)
        print(f"🖼️ User photo saved: {user_photo_path}")

        # Resize and enhance photo
        img = Image.open(user_photo_path).convert("RGB")
        img = img.resize((512, 512))
        enhanced_path = f"/tmp/enhanced_face_{uuid.uuid4().hex}.jpg"
        img.save(enhanced_path)
        print("✨ Face enhanced and resized")

        # Handle voice
        if user_audio:
            user_audio_path = f"/tmp/user_voice_{uuid.uuid4().hex}.wav"
            user_audio.save(user_audio_path)
            final_audio = user_audio_path
            print("🎤 User voice uploaded")
        else:
            # Generate AI voice automatically
            voice_id = "21m00Tcm4TlvDq8ikWAM"
            ai_audio_path = f"/tmp/ai_avatar_voice_{uuid.uuid4().hex}.mp3"
            ai_audio = generate(text=script_text, voice=voice_id, model="eleven_multilingual_v2")
            save(ai_audio, ai_audio_path)
            final_audio = ai_audio_path
            print("🧠 Generated AI voice for avatar")

        # Generate talking avatar using Wav2Lip (lip sync)
        final_avatar_video = f"/tmp/final_avatar_{uuid.uuid4().hex}.mp4"
        os.system(f"python3 scripts/wav2lip.py --face {enhanced_path} --audio {final_audio} --outfile {final_avatar_video}")

        # Add cinematic post-effects
        final_avatar_video = apply_cinematic_lighting(final_avatar_video, "portrait")
        final_avatar_video = apply_dynamic_camera_effects(final_avatar_video, "closeup")

        print(f"🎬 Avatar video created successfully: {final_avatar_video}")
        return jsonify({
            "message": "Avatar video created successfully!",
            "video_path": final_avatar_video
        }), 200

    except Exception as e:
        print(f"❌ Avatar generation failed: {e}")
        return jsonify({"error": str(e)}), 500

# ============================================
# 🎭 AI Expression Engine (Emotion-Based Animation)
# ============================================

import cv2
import numpy as np

def detect_emotion_from_text(script_text: str) -> str:
    """
    Basic emotion detector from script keywords.
    You can expand with a real NLP emotion model later.
    """
    text = script_text.lower()
    if any(word in text for word in ["sad", "cry", "lonely", "hurt", "pain"]):
        return "sad"
    elif any(word in text for word in ["happy", "joy", "smile", "fun", "laugh"]):
        return "happy"
    elif any(word in text for word in ["angry", "rage", "furious", "fight"]):
        return "angry"
    elif any(word in text for word in ["shock", "surprise", "wow", "amazing"]):
        return "surprised"
    else:
        return "neutral"


def apply_facial_expression(video_path: str, emotion: str) -> str:
    """
    Adds dynamic facial expressions based on emotion.
    Uses OpenCV filters and overlay for cinematic feel.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Failed to open video for emotion enhancement.")
        return video_path

    output_path = f"/tmp/emotion_video_{uuid.uuid4().hex}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"🎨 Applying emotion overlay: {emotion}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        overlay = frame.copy()

        if emotion == "happy":
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 30, frame)
        elif emotion == "sad":
            blue_tint = np.full_like(frame, (100, 120, 200))
            frame = cv2.addWeighted(blue_tint, 0.3, frame, 0.7, 0)
        elif emotion == "angry":
            red_tint = np.full_like(frame, (200, 80, 80))
            frame = cv2.addWeighted(red_tint, 0.4, frame, 0.6, 10)
        elif emotion == "surprised":
            bright = cv2.convertScaleAbs(frame, alpha=1.3, beta=25)
            frame = cv2.addWeighted(bright, 0.7, frame, 0.3, 0)

        out.write(frame)

    cap.release()
    out.release()
    print(f"✅ Emotion overlay applied successfully: {output_path}")
    return output_path

# ============================================
# 🌄 AI Scene Fusion Engine (Automatic Environment Generator)
# ============================================

import random

# Predefined cinematic theme library (URLs or cloud assets)
SCENE_BACKGROUNDS = {
    "jungle": [
        "https://cdn.pixabay.com/video/2023/03/10/158969-808529993_large.mp4",
        "https://cdn.pixabay.com/video/2021/09/08/90557-598460154_large.mp4"
    ],
    "space": [
        "https://cdn.pixabay.com/video/2023/05/21/162244-828607506_large.mp4",
        "https://cdn.pixabay.com/video/2023/02/03/149894-799048304_large.mp4"
    ],
    "temple": [
        "https://cdn.pixabay.com/video/2022/11/18/139572-773815509_large.mp4"
    ],
    "city": [
        "https://cdn.pixabay.com/video/2023/06/04/164536-835406787_large.mp4"
    ],
    "mountain": [
        "https://cdn.pixabay.com/video/2023/07/06/167154-846043081_large.mp4"
    ],
    "fire": [
        "https://cdn.pixabay.com/video/2023/08/09/169232-856291781_large.mp4"
    ],
    "ocean": [
        "https://cdn.pixabay.com/video/2023/05/03/160956-823260676_large.mp4"
    ],
    "rain": [
        "https://cdn.pixabay.com/video/2023/09/09/172200-867774999_large.mp4"
    ],
    "sky": [
        "https://cdn.pixabay.com/video/2023/10/10/175501-880036152_large.mp4"
    ]
}


def detect_scene_theme(script_text: str) -> str:
    """
    Auto-detect the best matching scene type from the script.
    """
    text = script_text.lower()
    if any(word in text for word in ["jungle", "forest", "animals"]):
        return "jungle"
    elif any(word in text for word in ["space", "planet", "galaxy", "star"]):
        return "space"
    elif any(word in text for word in ["temple", "god", "prayer"]):
        return "temple"
    elif any(word in text for word in ["mountain", "hill", "snow", "adventure"]):
        return "mountain"
    elif any(word in text for word in ["city", "street", "modern", "urban"]):
        return "city"
    elif any(word in text for word in ["fire", "anger", "battle"]):
        return "fire"
    elif any(word in text for word in ["ocean", "sea", "water", "boat"]):
        return "ocean"
    elif any(word in text for word in ["rain", "sad", "cry", "monsoon"]):
        return "rain"
    elif any(word in text for word in ["sky", "heaven", "freedom"]):
        return "sky"
    else:
        return "city"  # default theme

def add_auto_scene_background(script_text: str, video_path: str) -> str:
    """
    Merge cinematic background video with the generated character/voice video.
    """
    theme = detect_scene_theme(script_text)
    bg_video = random.choice(SCENE_BACKGROUNDS.get(theme, []))

    output_path = f"/tmp/scene_fusion_{uuid.uuid4().hex}.mp4"
    print(f"🌄 Auto-applying scene theme: {theme} ({bg_video})")

    # Merge with ffmpeg
    os.system(f"ffmpeg -y -i '{bg_video}' -i '{video_path}' -filter_complex '[0:v][1:v]overlay=(W-w)/2:(H-h)/2' -shortest {output_path}")

    print(f"✅ Scene fusion complete: {output_path}")
    return output_path

# ============================================
# 🎧 AI Sound Emotion Composer (ASEC Engine)
# ============================================

import random

# Predefined emotion → music & sound effect library
SOUND_LIBRARY = {
    "happy": [
        "https://cdn.pixabay.com/audio/2023/03/14/audio_507b06a1a5.mp3",
        "https://cdn.pixabay.com/audio/2023/03/20/audio_7a11f9f75b.mp3"
    ],
    "sad": [
        "https://cdn.pixabay.com/audio/2022/12/27/audio_d52a493b68.mp3",
        "https://cdn.pixabay.com/audio/2023/02/05/audio_846f2f7cdb.mp3"
    ],
    "angry": [
        "https://cdn.pixabay.com/audio/2023/04/22/audio_c7a4f0a8a1.mp3",
        "https://cdn.pixabay.com/audio/2023/05/01/audio_fa92f28dbf.mp3"
    ],
    "romantic": [
        "https://cdn.pixabay.com/audio/2022/11/11/audio_bfa348f33f.mp3"
    ],
    "motivational": [
        "https://cdn.pixabay.com/audio/2023/03/06/audio_1479c10fa3.mp3",
        "https://cdn.pixabay.com/audio/2023/01/17/audio_b14e8df24e.mp3"
    ],
    "nature": [
        "https://cdn.pixabay.com/audio/2023/04/17/audio_1848a6e2c8.mp3"
    ],
    "mystery": [
        "https://cdn.pixabay.com/audio/2023/02/12/audio_2b1b66fd0a.mp3"
    ]
}

# Auto map theme to sound category
THEME_TO_MOOD = {
    "jungle": "nature",
    "space": "mystery",
    "temple": "romantic",
    "fire": "angry",
    "ocean": "sad",
    "rain": "sad",
    "mountain": "motivational",
    "city": "motivational",
    "sky": "happy"
}


def add_emotion_music(script_text: str, video_path: str) -> str:
    """
    Add background music + emotion-based sound design automatically.
    """
    emotion = detect_emotion_from_text(script_text)
    theme = detect_scene_theme(script_text)

    # Choose suitable music track
    music_category = THEME_TO_MOOD.get(theme, emotion)
    sound_track = random.choice(SOUND_LIBRARY.get(music_category, SOUND_LIBRARY["motivational"]))

    output_path = f"/tmp/final_soundmix_{uuid.uuid4().hex}.mp4"

    print(f"🎧 Adding soundscape: {music_category} ({sound_track})")

    # Merge video + audio with ffmpeg
    os.system(f"ffmpeg -y -i '{video_path}' -i '{sound_track}' -filter_complex '[1:a]volume=0.35[a1];[0:a][a1]amix=inputs=2:duration=longest' -shortest {output_path}")

    print(f"✅ Sound fusion complete: {output_path}")
    return output_path

@app.route("/generate-sound-video", methods=["POST"])
def generate_sound_video():
    """
    Generates final cinematic video with theme + emotion + soundscape.
    """
    try:
        data = request.get_json(force=True)
        script_text = data.get("script_text", "")
        video_path = data.get("video_path", "")

        if not script_text.strip():
            return jsonify({"error": "Missing script_text"}), 400
        if not os.path.exists(video_path):
            return jsonify({"error": "Invalid video path"}), 400

        final_sound_video = add_emotion_music(script_text, video_path)

        return jsonify({
            "message": "Final cinematic video with emotion-based sound added!",
            "video_path": final_sound_video
        }), 200

    except Exception as e:
        print(f"❌ Sound generation failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/generate-scene-video", methods=["POST"])
def generate_scene_video():
    """
    Automatically builds a cinematic AI video with auto-detected background theme.
    """
    try:
        data = request.get_json(force=True)
        script_text = data.get("script_text", "")
        video_path = data.get("video_path", "")

        if not script_text.strip():
            return jsonify({"error": "Missing script_text"}), 400
        if not os.path.exists(video_path):
            return jsonify({"error": "Invalid video path"}), 400

        fused_video = add_auto_scene_background(script_text, video_path)

        return jsonify({
            "message": "Cinematic scene video generated successfully!",
            "theme": detect_scene_theme(script_text),
            "video_path": fused_video
        }), 200

    except Exception as e:
        print(f"❌ Scene fusion failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/generate-expressive-video", methods=["POST"])
def generate_expressive_video():
    """
    Create a cinematic video with dynamic face emotion effects.
    """
    try:
        data = request.get_json(force=True)
        script_text = data.get("script_text", "")
        video_path = data.get("video_path", "")

        if not script_text.strip():
            return jsonify({"error": "Script text is required"}), 400

        if not os.path.exists(video_path):
            return jsonify({"error": "Invalid or missing video path"}), 400

        # Detect emotion and apply effect
        emotion = detect_emotion_from_text(script_text)
        expressive_video = apply_facial_expression(video_path, emotion)

        print(f"🎭 Expressive video created successfully: {expressive_video}")
        return jsonify({
            "message": "Expressive cinematic video created!",
            "emotion": emotion,
            "video_path": expressive_video
        }), 200

    except Exception as e:
        print(f"❌ Emotion video generation failed: {e}")
        return jsonify({"error": str(e)}), 500

# -------------------- API Endpoint: Generate Video --------------------
@app.route("/generate-video", methods=["POST"])
def generate_video_api():
    """
    API endpoint to generate cinematic AI video from a text script.
    Automatically creates scenes, voices, and visuals.
    """
    try:
        data = request.get_json(force=True)
        script_text = data.get("script_text", "")
        if not script_text.strip():
            return jsonify({"error": "Missing script_text"}), 400

        # 🔥 Run the full cinematic AI generation pipeline
        output_path = build_ai_composed_video(script_text)

        if output_path:
            # Success response
            return jsonify({
                "status": "success",
                "message": "Cinematic AI video generated successfully.",
                "video_url": output_path
            }), 200
        else:
            return jsonify({"error": "Video generation failed"}), 500

    except Exception as e:
        log.error(f"API Error in /generate-video: {e}")
        return jsonify({"error": str(e)}), 500

# -------------------- Requirements & .env template --------------------
REQUIREMENTS_TXT = """
Flask
Flask-SQLAlchemy
Flask-Cors
PyJWT
moviepy
gTTS
requests
boto3
google-cloud-storage
firebase-admin
python-dotenv
razorpay
"""

ENV_TEMPLATE = """
# Visora Backend .env template (copy to .env and fill values)
SECRET_KEY=change_this_secret
DATABASE_URL=sqlite:///visora.db
OPENAI_API_KEY=your_openai_key_or_leave_empty
OPENAI_MODEL=gpt-4o-mini
AWS_S3_BUCKET=your-bucket
AWS_ACCESS_KEY_ID=your-aws-key
AWS_SECRET_ACCESS_KEY=your-aws-secret
AWS_S3_REGION=us-east-1
FIREBASE_CRED_JSON=/path/to/firebase-service-account.json
FIREBASE_BUCKET=your-firebase-bucket.appspot.com
PAYPAL_CLIENT_ID=your-paypal-client-id
PAYPAL_SECRET=your-paypal-secret
PAYPAL_SANDBOX=1
RAZORPAY_KEY_ID=your-razorpay-id
RAZORPAY_KEY_SECRET=your-razorpay-secret
CLEANUP_INTERVAL_SEC=3600
FILE_RETENTION_DAYS=7
"""

# -------------------- Deployment & Run instructions (Hindi) --------------------
DEPLOY_INSTRUCTIONS = """
Visora Backend - Run / Deploy Instructions (Hindi)

1) लोकल टेस्ट:
   - Python >= 3.9 चाहिए
   - एक virtualenv बनाओ:
       python -m venv venv
       source venv/bin/activate   (Linux/Mac)  OR  venv\\Scripts\\activate (Windows)
   - requirements.txt install करो (create एक file और ऊपर REQUIREMENTS_TXT content paste करो):
       pip install -r requirements.txt
   - .env फाइल बनाओ और ENV_TEMPLATE के अनुसार भरा करो (optional keys जब नहीं हों तो कुछ फीचर्स degraded mode में चलेंगे)
   - Run:
       python visora_backend.py
   - डिफ़ॉल्ट port 5000 खुलेगा. Render पर पोर्ट env VAR से बदलना होगा (Render sets PORT env var).

2) Render पर Deploy:
   - Git repo बनाओ, visora_backend.py और requirements.txt commit करो.
   - Render के New Web Service पे जाकर GitHub repo connect करो.
   - Start Command: python visora_backend.py (या gunicorn -w 4 visora_backend:app)
   - Env vars में अपनी keys डालो (.env values)

3) Firebase Compatible:
   - अगर Firebase Storage करना है तो FIREBASE_CRED_JSON path और FIREBASE_BUCKET set करो.
   - firebase_admin lib install करो (requirements में है).

4) Notes:
   - MoviePy heavy है; ffmpeg भी install होना चाहिए (system-level). Render custom build में ffmpeg availability चेक करो.
   - OpenAI integration के लिए OPENAI_API_KEY डालो.
   - Payments: Razorpay requires keys; PayPal requires client_id & secret.

"""

# Print final helpful hints
print("✅ Requirements sample (copy content to requirements.txt):\n", REQUIREMENTS_TXT)
print("✅ .env template (copy to .env):\n", ENV_TEMPLATE)
print("✅ Deployment instructions:\n", DEPLOY_INSTRUCTIONS)
