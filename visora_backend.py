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

# ----------------------------------------------------------------------------
# AUTH ENDPOINTS
# ----------------------------------------------------------------------------
@app.route("/register", methods=["POST"])
def register():
    data = request.get_json(force=True)
    email, password = data.get("email"), data.get("password")
    if not email or not password:
        return jsonify({"error": "Missing credentials"}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({"error": "User already exists"}), 409
    u = User(email=email, name=data.get("name"), password=password)
    db.session.add(u)
    db.session.commit()
    return jsonify({"message": "User registered", "token": token_create(email)})

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(force=True)
    email, password = data.get("email"), data.get("password")
    u = User.query.filter_by(email=email).first()
    if not u or u.password != password:
        return jsonify({"error": "Invalid credentials"}), 401
    return jsonify({"token": token_create(email), "user": {"email": u.email, "plan": u.plan, "credits": u.credits}})

@app.route("/profile", methods=["GET"])
def profile():
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    decoded = token_verify(token)
    if not decoded:
        return jsonify({"error": "Invalid token"}), 401
    u = User.query.filter_by(email=decoded["email"]).first()
    return jsonify({
        "email": u.email,
        "name": u.name,
        "plan": u.plan,
        "credits": u.credits,
        "country": u.country
    })

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

# -------------------- Emotion-Based Cinematic Tone Enhancer --------------------
import textblob
from moviepy.editor import vfx

def analyze_emotion_from_text(text: str) -> str:
    """
    Detect emotion (happy, sad, angry, excited) from text using polarity.
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


def apply_emotion_filter(clip, emotion: str):
    """
    Apply color tone, zoom, and brightness based on emotion.
    """
    try:
        if emotion == "happy":
            clip = clip.fx(vfx.colorx, 1.2)  # brighter
        elif emotion == "excited":
            clip = clip.fx(vfx.colorx, 1.4).fx(vfx.speedx, 1.1)
        elif emotion == "sad":
            clip = clip.fx(vfx.colorx, 0.7).fx(vfx.lum_contrast, lum=20, contrast=40)
        elif emotion == "angry":
            clip = clip.fx(vfx.colorx, 0.9).fx(vfx.lum_contrast, lum=-30, contrast=90)
        else:
            clip = clip.fx(vfx.colorx, 1.0)
        return clip
    except Exception as e:
        log.warning(f"Emotion filter failed: {e}")
        return clip


def build_emotion_cinematic_video(script_text: str):
    """
    Combine emotion detection + avatar + voice to build cinematic story.
    """
    segments = detect_characters_from_script(script_text)
    clips = []

    for seg in segments:
        name = seg["character"]
        voice_type = seg["voice_type"]
        text = seg["text"]
        emotion = analyze_emotion_from_text(text)
        avatar_url = fetch_avatar_for_character(name, voice_type)

        log.info(f"🎬 Scene: {name} ({emotion}) → {text}")

        try:
            img_clip = ImageClip(avatar_url).resize((720, 1280))
            voice_path = generate_character_voice(name, text)
            audio_clip = AudioFileClip(voice_path)
            dur = max(audio_clip.duration, 3)

            # Apply cinematic + emotion filters
            img_clip = cinematic_motion(img_clip)
            img_clip = apply_emotion_filter(img_clip, emotion)

            final_clip = img_clip.set_duration(dur).set_audio(audio_clip)
            clips.append(final_clip)
        except Exception as e:
            log.warning(f"Scene generation failed for {name}: {e}")

    if clips:
        final_video = concatenate_videoclips(clips, method="compose")
        output_path = OUTPUT_FOLDER / f"emotion_story_{uuid.uuid4().hex}.mp4"
        final_video.write_videofile(str(output_path), fps=24)
        log.info(f"🎞 Emotion-based cinematic video created: {output_path}")
        return str(output_path)
    else:
        log.warning("⚠️ No clips generated.")
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
    return jsonify({"message": "cinematic video created", "file": v.file_path, "video_id": v.id})

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
