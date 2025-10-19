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
