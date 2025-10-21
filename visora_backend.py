#!/usr/bin/env python3
"""
Visora Backend v2.0 - Universal AI Production Server
----------------------------------------------------
Author: Aimantuvya & GPT-5
Description:
  - AI-based cinematic video generator backend
  - Flask + UCVE (Universal Cinematic Video Engine)
  - Firebase + S3 + SQLite hybrid sync system
  - Render + GitHub + Termux compatible
"""

from flask import Flask, request, jsonify
import os, uuid, datetime, json, logging as log
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, TextClip, vfx
from typing import Optional
from time import time
import random
import numpy as np
import librosa

app = Flask(__name__)

# 🧠 In-memory user rate tracking
user_requests = {}

def rate_limit(user_ip, limit=100, period=60):
    """
    Custom limiter: allows up to <limit> requests per <period> seconds per IP.
    """
    now = time()
    # Purane timestamps remove karo
    user_requests[user_ip] = [t for t in user_requests.get(user_ip, []) if now - t < period]

    if len(user_requests[user_ip]) >= limit:
        return False  # limit exceeded
    user_requests[user_ip].append(now)
    return True


@app.before_request
def limit_requests():
    ip = request.remote_addr or "unknown"
    if not rate_limit(ip):
        return jsonify({"error": "Too many requests, please slow down!"}), 429


# 🧩 Base Configuration
BASE_DIR = os.getcwd()
RENDER_PATH = os.path.join(BASE_DIR, "renders")
os.makedirs(RENDER_PATH, exist_ok=True)

# 🧩 Base Configuration
BASE_DIR = os.getcwd()
RENDER_PATH = os.path.join(BASE_DIR, "renders")
os.makedirs(RENDER_PATH, exist_ok=True)

# ------------------------------
# 🧠 AI Assistant (Placeholder)
# ------------------------------
@app.route("/assistant", methods=["POST"])
def assistant():
    data = request.get_json() or {}
    tone = data.get("tone", "motivational")
    prompt = data.get("query", "")
    if not prompt:
        return jsonify({"error": "Missing query"}), 400

    # Basic tone-based replies (expand later)
    if tone == "motivational":
        reply = f"🔥 Keep going! {prompt[:80]}..."
    elif tone == "funny":
        reply = f"😂 Here's a fun start: {prompt[:80]}..."
    else:
        reply = f"💡 Idea: {prompt[:80]}..."

    return jsonify({"response": reply, "status": "ok"})

# ===============================================================
# 🎬 UCVE: Cinematic Video Generator (with short/long selector)
# ===============================================================

@app.route("/generate_video", methods=["POST"])
def generate_video():
    data = request.get_json() or {}
    script_text = data.get("script", "No script provided.")
    video_type = data.get("video_type", "short").lower()  # user input or default short
    video_id = str(uuid.uuid4())

    # 🎞️ Choose config based on video type
    if video_type == "short":
        config = {
            "duration_limit": 60,
            "transition_speed": "fast",
            "scene_complexity": "medium",
            "music_mode": "dynamic"
        }
    else:
        config = {
            "duration_limit": 600,
            "transition_speed": "cinematic",
            "scene_complexity": "high",
            "music_mode": "deep_ambient"
        }

    try:
        out_file = os.path.join(RENDER_PATH, f"{video_id}.txt")

        # 🎥 Simulated render process (placeholder – actual render will replace)
        with open(out_file, "w") as f:
            f.write(f"Rendered {video_type} video for: {script_text}\n")
            f.write(f"Applied Config: {json.dumps(config, indent=2)}\n")

        log.info(f"{video_type.capitalize()} video rendered successfully!")

        return jsonify({
            "status": "✅ success",
            "video_id": video_id,
            "video_type": video_type,
            "config_used": config,
            "file": out_file,
            "message": f"{video_type.capitalize()} cinematic render completed successfully."
        })

    except Exception as e:
        log.exception("Video generation failed")
        return jsonify({
            "status": "❌ error",
            "message": str(e)
        }), 500

# ------------------------------
# 💾 Health Check API
# ------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "time": datetime.datetime.now().isoformat(),
        "server": "Visora Backend v2.0"
    })

# ====================================================
# 🎬 UCVE: Universal Cinematic Video Engine v2.0
# ====================================================
import random
import textblob

# Emotion-based cinematic tone detection
def analyze_emotion(script_text: str) -> str:
    try:
        blob = textblob.TextBlob(script_text)
        sentiment = blob.sentiment.polarity
        if sentiment > 0.3:
            return "happy"
        elif sentiment < -0.3:
            return "sad"
        else:
            return "neutral"
    except Exception as e:
        log.exception("Emotion analysis failed")
        return "neutral"

# Cinematic background soundscape mapping
CINEMATIC_SOUNDS = {
    "happy": ["forest_birds.mp3", "calm_wind.mp3", "soft_piano.mp3"],
    "sad": ["rain_ambience.mp3", "soft_violin.mp3", "dark_calm.mp3"],
    "neutral": ["city_ambience.mp3", "daylight_soft.mp3"]
}

# Simulated ambient soundscape (placeholder)
def add_ambient_sound(scene_path: str, mood: str) -> str:
    amb = random.choice(CINEMATIC_SOUNDS.get(mood, CINEMATIC_SOUNDS["neutral"]))
    log.info(f"Ambient sound added: {amb}")
    return f"{scene_path} + {amb}"

# Dynamic camera effect simulation
def apply_dynamic_camera(scene_path: str, mood: str) -> str:
    effects = {
        "happy": "smooth_pan_zoom",
        "sad": "slow_drift",
        "neutral": "steady_focus"
    }
    effect = effects.get(mood, "steady_focus")
    log.info(f"Applied camera motion: {effect}")
    return f"{scene_path} + {effect}"

# Cinematic lighting postprocessor
def apply_cinematic_lighting(scene_path: str, mood: str) -> str:
    lighting = {
        "happy": "bright_warm_tone",
        "sad": "cool_dark_tone",
        "neutral": "balanced_soft_tone"
    }
    tone = lighting.get(mood, "balanced_soft_tone")
    log.info(f"Applied lighting tone: {tone}")
    return f"{scene_path} + {tone}"

# Main cinematic processor
def generate_cinematic_scene(script_text: str) -> str:
    try:
        mood = analyze_emotion(script_text)
        base_scene = f"scene_base_{uuid.uuid4().hex[:8]}"
        log.info(f"🎭 Detected mood: {mood}")

        # Step 1: Dynamic Camera
        scene_with_camera = apply_dynamic_camera(base_scene, mood)

        # Step 2: Add Ambient Sound
        scene_with_sound = add_ambient_sound(scene_with_camera, mood)

        # Step 3: Apply Lighting
        final_scene = apply_cinematic_lighting(scene_with_sound, mood)

        return f"{final_scene}_rendered_final.mp4"

    except Exception as e:
        log.exception("UCVE Scene generation failed")
        return "error_scene.mp4"


# Extend Flask endpoint to use UCVE
@app.route("/ucve_generate", methods=["POST"])
def ucve_generate():
    data = request.get_json() or {}
    script = data.get("script", "")
    if not script:
        return jsonify({"error": "Missing script text"}), 400

    try:
        result_file = generate_cinematic_scene(script)
        return jsonify({
            "status": "success",
            "file": result_file,
            "mood": analyze_emotion(script)
        })
    except Exception as e:
        log.exception("UCVE endpoint failed")
        return jsonify({"status": "error", "message": str(e)})

# ====================================================
# 🔊 TTS + Auto-dub module (ElevenLabs primary, gTTS fallback)
# ====================================================
import os
import io
import uuid
import logging as log
from pathlib import Path

# Try optional imports
MOVIEPY_AVAILABLE = False
ELEVEN_AVAILABLE = False
GTTS_AVAILABLE = False
PYDUB_AVAILABLE = False

try:
    from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
    MOVIEPY_AVAILABLE = True
except Exception as e:
    log.info("moviepy not available: %s", e)

try:
    # optional official SDK may be present
    from elevenlabs import generate as eleven_generate, save as eleven_save, set_api_key as eleven_set_api_key
    ELEVEN_AVAILABLE = True
except Exception:
    try:
        import requests
        ELEVEN_AVAILABLE = True  # we'll call REST if SDK not present
    except Exception:
        ELEVEN_AVAILABLE = False

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except Exception:
    GTTS_AVAILABLE = False

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except Exception:
    PYDUB_AVAILABLE = False

# output folder
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def synthesize_tts(text: str, voice: str = "alloy", filename: str = None) -> str:
    """
    Generate speech audio from `text`.
    Priorities:
      1) ElevenLabs SDK / REST (if ELEVENLABS_API_KEY provided)
      2) gTTS fallback
    Returns local file path to generated .mp3 (or .wav) or raises Exception.
    """
    if not filename:
        filename = OUTPUT_DIR / f"tts_{uuid.uuid4().hex[:8]}.mp3"
    else:
        filename = Path(filename)

    # Try ElevenLabs if API key present
    ele_key = os.getenv("ELEVENLABS_API_KEY") or os.getenv("ELEVENLABS_KEY")
    if ele_key and ELEVEN_AVAILABLE:
        try:
            # If SDK available, use it; otherwise fallback to simple REST
            try:
                if 'eleven_set_api_key' in globals():
                    eleven_set_api_key(ele_key)
                    audio_bytes = eleven_generate(text=text, voice=voice, model="eleven_monolingual_v1")
                    # SDK may return bytes-like object or a path; handle bytes-like
                    if isinstance(audio_bytes, (bytes, bytearray)):
                        with open(filename, "wb") as f:
                            f.write(audio_bytes)
                    else:
                        # if SDK provided a file-like save function
                        eleven_save(audio_bytes, str(filename))
                else:
                    # REST approach (generic)
                    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice}"
                    headers = {
                        "xi-api-key": ele_key,
                        "Content-Type": "application/json"
                    }
                    payload = {"text": text, "voice": voice}
                    resp = requests.post(url, json=payload, headers=headers, stream=True, timeout=60)
                    resp.raise_for_status()
                    with open(filename, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                log.info("TTS generated via ElevenLabs -> %s", filename)
                return str(filename)
            except Exception as e:
                log.warning("ElevenLabs TTS failed: %s", e)
                # fallthrough to gTTS
        except Exception as e:
            log.warning("ElevenLabs generation error: %s", e)

    # gTTS fallback
    if GTTS_AVAILABLE:
        try:
            tts = gTTS(text=text, lang="en", slow=False)
            tts.save(str(filename))
            log.info("TTS generated via gTTS -> %s", filename)
            return str(filename)
        except Exception as e:
            log.exception("gTTS failed: %s", e)

    # If nothing worked, raise
    raise RuntimeError("No TTS backend available (set ELEVENLABS_API_KEY or install gTTS)")

def attach_audio_to_video(video_path: str, audio_path: str, out_path: str = None) -> str:
    """
    Attach audio (audio_path) to existing video (video_path) using MoviePy.
    Returns output file path.
    """
    if not MOVIEPY_AVAILABLE:
        raise RuntimeError("moviepy required to attach audio to video")

    video_path = str(video_path)
    audio_path = str(audio_path)

    if not out_path:
        out_path = OUTPUT_DIR / f"dubbed_{uuid.uuid4().hex[:8]}.mp4"
    else:
        out_path = Path(out_path)

    try:
        vclip = VideoFileClip(video_path)
        aclip = AudioFileClip(audio_path)

        # If audio shorter or longer, we set audio to video duration by subclip or loop
        if aclip.duration < vclip.duration:
            # loop audio if short (simple approach using pydub if available)
            if PYDUB_AVAILABLE:
                # create loop using pydub for better quality/format handling
                sound = AudioSegment.from_file(audio_path)
                times = int(vclip.duration / (sound.duration_seconds or 1)) + 1
                combined = sound * times
                tmp_loop = OUTPUT_DIR / f"looped_{uuid.uuid4().hex[:8]}.mp3"
                combined[:int(vclip.duration * 1000)].export(str(tmp_loop), format="mp3")
                aclip = AudioFileClip(str(tmp_loop))
            else:
                # simple subclip (audio will stop before video ends)
                pass
        elif aclip.duration > vclip.duration:
            aclip = aclip.subclip(0, vclip.duration)

        final = vclip.set_audio(aclip)
        final.write_videofile(str(out_path), codec="libx264", audio_codec="aac", temp_audiofile=str(OUTPUT_DIR / "temp-audio.m4a"), remove_temp=True, threads=2, logger=None)
        # close clips
        try:
            final.close()
            vclip.close()
            aclip.close()
        except Exception:
            pass

        log.info("Audio attached -> %s", out_path)
        return str(out_path)

    except Exception as e:
        log.exception("Attaching audio failed: %s", e)
        raise

# Flask endpoint: synthesize TTS and attach to given video (or UCVE output)
@app.route("/synthesize_and_attach", methods=["POST"])
def synthesize_and_attach():
    """
    POST JSON:
      {
        "script": "text to speak",
        "video_path": "optional existing video path (local relative path) or 'ucve' to auto-generate",
        "voice": "optional voice name",
      }
    Returns: {status, file}
    """
    data = request.get_json() or {}
    script = data.get("script", "")
    video_path = data.get("video_path", "")
    voice = data.get("voice", "alloy")

    if not script:
        return jsonify({"error": "script required"}), 400

    try:
        # If user requested UCVE generate first, call internal function
        if video_path == "ucve" or not video_path:
            # use UCVE generate; re-use generate_cinematic_scene if present
            if "generate_cinematic_scene" in globals():
                video_path = generate_cinematic_scene(script)
            else:
                return jsonify({"error": "no video path provided and UCVE generator not found"}), 400

        # generate tts
        tts_file = synthesize_tts(script, voice=voice)

        # attach
        final_file = attach_audio_to_video(video_path, tts_file)

        return jsonify({"status": "success", "file": final_file})

    except Exception as e:
        log.exception("synthesize_and_attach failed")
        return jsonify({"status": "error", "message": str(e)}), 500

# ====================================================
# 🎭 UCVE v3: Multi-Character Dialogue + Voice Engine
# ====================================================
import re

# Character voice map (expandable / dynamic)
CHARACTER_VOICES = {
    "narrator": "alloy",
    "tiger": "adam",
    "monkey": "bella",
    "human": "antoni",
}

def detect_dialogues(script_text: str):
    """
    Detect dialogues in format:
    Tiger: The jungle is mine.
    Monkey: Not until you climb that tree!
    """
    lines = script_text.splitlines()
    dialogues = []
    for line in lines:
        m = re.match(r"^([A-Za-z]+):\s*(.+)$", line.strip())
        if m:
            name, text = m.groups()
            dialogues.append((name.lower(), text.strip())
    # fallback: if no dialogues found, treat all as narrator
    return jsonify({"error": "No dialogues found"}), 400
    return jsonify({"error": "No dialogues found"}), 400
    if not dialogues:
        dialogues = [("narrator", script_text)]
    return dialogues

def generate_multi_voice_scene(script_text: str):
    """
    Generate multi-character TTS audio sequence and combine.
    Returns path to final combined dubbed video.
    """
    try:
        dialogues = detect_dialogues(script_text)
        mood = analyze_emotion(script_text)
        log.info(f"🎭 Detected {len(dialogues)} dialogues, mood={mood}")

        # Step 1: UCVE base scene render
        base_video = generate_cinematic_scene(script_text)

        # Step 2: Generate voice clips for each dialogue
        audio_clips = []
        for idx, (char, line) in enumerate(dialogues):
            voice = CHARACTER_VOICES.get(char, "alloy")
            log.info(f"Generating voice for {char} → {voice}")
            clip_file = synthesize_tts(line, voice=voice)
            audio_clips.append(clip_file)

        # Step 3: Merge all audio clips sequentially
        if PYDUB_AVAILABLE:
            from pydub import AudioSegment
            combined = AudioSegment.empty()
            for clip in audio_clips:
                combined += AudioSegment.from_file(clip)
            merged_path = OUTPUT_DIR / f"multi_voice_{uuid.uuid4().hex[:8]}.mp3"
            combined.export(str(merged_path), format="mp3")
        else:
            merged_path = audio_clips[0]  # fallback to first voice

        # Step 4: Attach combined audio to cinematic scene
        final_video = attach_audio_to_video(base_video, str(merged_path))

        return final_video

    except Exception as e:
        log.exception("Multi-voice scene generation failed")
        return "multi_voice_error.mp4"


# Flask endpoint for multi-voice scene generator
@app.route("/multi_voice_scene", methods=["POST"])
def multi_voice_scene():
    """
    POST JSON:
      {
        "script": "Tiger: Roar!\nMonkey: Hahaha you can’t catch me!"
      }
    Returns:
      {
        "status": "success",
        "file": "<path to final video>",
        "voices_used": ["tiger", "monkey"]
      }
    """
    data = request.get_json() or {}
    script = data.get("script", "")
    if not script:
        return jsonify({"error": "Script text required"}), 400

    try:
        final_file = generate_multi_voice_scene(script)
        detected_chars = [c for c, _ in detect_dialogues(script)]
        return jsonify({
            "status": "success",
            "file": final_file,
            "voices_used": detected_chars
        })
    except Exception as e:
        log.exception("multi_voice_scene endpoint failed")
        return jsonify({"status": "error", "message": str(e)}), 500

# ====================================================
# 💬 UCVE v4: Subtitle Generator + Translator + Caption Burn-in
# ====================================================
from moviepy.editor import TextClip, CompositeVideoClip
from textblob import TextBlob

def generate_subtitles(script_text: str, lang_target: str = "en"):
    """
    Generate list of (text, start, end) tuples for subtitles from script text.
    Simple per-line segmentation.
    """
    lines = script_text.strip().splitlines()
    subs = []
    start_time = 0.0
    dur_per_line = 3.5  # seconds per line (approx)
    for line in lines:
        if not line.strip():
            continue
        end_time = start_time + dur_per_line
        subs.append((line.strip(), start_time, end_time))
        start_time = end_time
    return subs

def translate_text(text: str, target_lang: str = "en"):
    """
    Translate using TextBlob (Google Translate backend).
    """
    try:
        if target_lang.lower() in ("en", "english"):
            return text
        blob = TextBlob(text)
        translated = str(blob.translate(to=target_lang))
        return translated
    except Exception as e:
        log.warning("Translation failed for %s: %s", text, e)
        return text

def burn_subtitles_to_video(video_path: str, script_text: str, lang_target: str = "en"):
    """
    Overlay cinematic subtitles (translated if requested) on video using MoviePy.
    Returns path to new subtitled mp4.
    """
    if not MOVIEPY_AVAILABLE:
        raise RuntimeError("moviepy required for subtitle overlay")

    subs = generate_subtitles(script_text, lang_target)
    vclip = VideoFileClip(video_path)

    subtitle_clips = []
    for text, start, end in subs:
        translated_text = translate_text(text, target_lang=lang_target)
        txt_clip = TextClip(
            translated_text,
            fontsize=38,
            color='yellow',
            font='Arial-Bold',
            stroke_color='black',
            stroke_width=2,
            method='caption',
            size=(vclip.w - 100, None)
        ).set_position(('center', 'bottom')).set_duration(end - start).set_start(start)
        subtitle_clips.append(txt_clip)

    final = CompositeVideoClip([vclip, *subtitle_clips])
    out_path = OUTPUT_DIR / f"subtitled_{uuid.uuid4().hex[:8]}.mp4"
    final.write_videofile(str(out_path), codec="libx264", audio_codec="aac", threads=2, logger=None)

    # Cleanup
    try:
        final.close()
        vclip.close()
    except Exception:
        pass

    log.info("Subtitles burned in -> %s", out_path)
    return str(out_path)


# Flask endpoint for subtitle translation + burn-in
@app.route("/generate_subtitles", methods=["POST"])
def subtitles_endpoint():
    """
    POST JSON:
      {
        "video_path": "path/to/video.mp4",
        "script": "Your full script text here",
        "lang_target": "hi"
      }
    """
    data = request.get_json() or {}
    script = data.get("script", "")
    video_path = data.get("video_path", "")
    lang_target = data.get("lang_target", "en")

    if not video_path or not script:
        return jsonify({"error": "video_path and script required"}), 400

    try:
        output = burn_subtitles_to_video(video_path, script, lang_target)
        return jsonify({
            "status": "success",
            "file": output,
            "language": lang_target
        })
    except Exception as e:
        log.exception("Subtitle generation failed")
        return jsonify({"status": "error", "message": str(e)}), 500

# ====================================================
# 🎶 UCVE v5: Emotion-based Background Music Composer
# ====================================================
import numpy as np
from pydub.generators import Sine
from pydub import AudioSegment

def compose_emotion_music(mood: str, duration: float = 20.0) -> str:
    """
    Compose simple emotion-based instrumental track.
    Uses sine waves + harmony tones to simulate emotion-based soundtrack.
    """
    base_freqs = {
        "happy": [440, 660, 880],  # A major (bright)
        "sad": [220, 330, 440],    # A minor (dark)
        "neutral": [392, 523, 659] # G major (soft)
    }
    freqs = base_freqs.get(mood, base_freqs["neutral"])

    segment = AudioSegment.silent(duration=0)
    for f in freqs:
        wave = Sine(f).to_audio_segment(duration=int(duration * 1000))
        segment = segment.overlay(wave - 6)  # lower volume to mix

    out_path = OUTPUT_DIR / f"music_{mood}_{uuid.uuid4().hex[:6]}.mp3"
    segment.export(str(out_path), format="mp3")
    log.info(f"🎵 Composed {mood} music: {out_path}")
    return str(out_path)


def attach_music_to_video(video_path: str, mood: str = "neutral"):
    """
    Generate background music and attach to video.
    """
    try:
        vclip = VideoFileClip(video_path)
        duration = vclip.duration
        bg_music = compose_emotion_music(mood, duration)
        return attach_audio_to_video(video_path, bg_music)
    except Exception as e:
        log.exception("Failed to attach music")
        return video_path


# Flask endpoint for automatic emotion-based music + video merge
@app.route("/auto_music_scene", methods=["POST"])
def auto_music_scene():
    """
    POST JSON:
      {
        "script": "Your scene text"
      }
    """
    data = request.get_json() or {}
    script = data.get("script", "")
    if not script:
        return jsonify({"error": "Script text required"}), 400

    try:
        # Step 1: Generate cinematic scene
        video = generate_cinematic_scene(script)

        # Step 2: Detect emotion & compose music
        mood = analyze_emotion(script)
        final_file = attach_music_to_video(video, mood)

        return jsonify({
            "status": "success",
            "mood": mood,
            "file": final_file
        })

    except Exception as e:
        log.exception("Auto music generation failed")
        return jsonify({"status": "error", "message": str(e)}), 500

# ====================================================
# ☁️ UCVE v6: Firebase Cloud Upload + Auto Sync
# ====================================================
import firebase_admin
from firebase_admin import credentials, storage

FIREBASE_BUCKET = None

def init_firebase():
    """
    Initialize Firebase app if not already done.
    """
    global FIREBASE_BUCKET
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate("visora-firebase-key.json")
            firebase_admin.initialize_app(cred, {
                "storageBucket": os.getenv("FIREBASE_BUCKET_NAME", "your-bucket-name.appspot.com")
            })
        FIREBASE_BUCKET = storage.bucket()
        log.info("🔥 Firebase initialized successfully.")
    except Exception as e:
        log.exception("Firebase initialization failed: %s", e)
        FIREBASE_BUCKET = None


def upload_to_firebase(local_path: str, remote_path: str = None) -> Optional[str]:
    """
    Upload file to Firebase Storage and return public URL.
    """
    if not FIREBASE_BUCKET:
        init_firebase()
    if not FIREBASE_BUCKET:
        raise RuntimeError("Firebase not initialized")

    local_file = Path(local_path)
    if not remote_path:
        remote_path = f"visora_uploads/{local_file.name}"

    blob = FIREBASE_BUCKET.blob(remote_path)
    blob.upload_from_filename(str(local_file))
    blob.make_public()

    public_url = blob.public_url
    log.info(f"☁️ Uploaded {local_file} → {public_url}")
    return public_url


@app.route("/cloud_upload", methods=["POST"])
def cloud_upload():
    """
    POST JSON:
      {
        "file_path": "outputs/video.mp4"
      }
    """
    data = request.get_json() or {}
    file_path = data.get("file_path", "")
    if not file_path:
        return jsonify({"error": "file_path required"}), 400

    try:
        url = upload_to_firebase(file_path)
        return jsonify({"status": "success", "file_url": url})
    except Exception as e:
        log.exception("Firebase upload failed")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/auto_sync", methods=["POST"])
def auto_sync():
    """
    Automatically upload last rendered video to Firebase and return cloud link.
    """
    data = request.get_json() or {}
    script = data.get("script", "")
    if not script:
        return jsonify({"error": "Script required"}), 400

    try:
        # Step 1: Generate cinematic scene
        local_video = generate_cinematic_scene(script)
        # Step 2: Upload to Firebase
        url = upload_to_firebase(local_video)
        return jsonify({
            "status": "success",
            "cloud_url": url,
            "file": local_video
        })
    except Exception as e:
        log.exception("Auto sync failed")
        return jsonify({"status": "error", "message": str(e)}), 500

# ====================================================
# 🧠 UCVE v7: AI Actor Generator (Virtual Human + Lip Sync)
# ====================================================
from PIL import Image, ImageDraw, ImageFont
import math

ACTOR_MODELS = {
    "male": "actors/male_default.png",
    "female": "actors/female_default.png",
    "child": "actors/child_default.png",
    "old": "actors/old_default.png"
}

def generate_actor_image(actor_type: str = "male", text: str = "") -> str:
    """
    Generate or load an actor face based on type (male/female/child/old)
    and overlay name or role text.
    """
    base_path = ACTOR_MODELS.get(actor_type, ACTOR_MODELS["male"])
    img = Image.open(base_path).convert("RGBA")

    # Overlay text
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.text((20, img.height - 40), f"{actor_type.title()} Role", fill=(255, 255, 0), font=font)

    out_path = OUTPUT_DIR / f"actor_{actor_type}_{uuid.uuid4().hex[:6]}.png"
    img.save(out_path)
    log.info(f"🎭 Actor generated: {out_path}")
    return str(out_path)


def simulate_lip_sync(image_path: str, audio_path: str) -> str:
    """
    Simple AI-simulated lip-sync (frame variation using MoviePy).
    """
    if not MOVIEPY_AVAILABLE:
        raise RuntimeError("MoviePy required for lip-sync")

    from moviepy.editor import ImageSequenceClip, AudioFileClip
    img = Image.open(image_path)
    frames = []

    # Create fake lip motion frames
    for i in range(6):
        mod = img.copy()
        draw = ImageDraw.Draw(mod)
        offset = 3 * math.sin(i)
        draw.ellipse((90, 250 + offset, 150, 260 + offset), fill=(255, 180, 180))
        temp_path = TMP_FOLDER / f"frame_{i}.png"
        mod.save(temp_path)
        frames.append(str(temp_path))

    clip = ImageSequenceClip(frames, fps=6)
    aclip = AudioFileClip(audio_path)
    final = clip.set_audio(aclip)

    out_path = OUTPUT_FOLDER / f"actor_lipsync_{uuid.uuid4().hex[:8]}.mp4"
    final.write_videofile(str(out_path), codec="libx264", audio_codec="aac", logger=None)

    final.close()
    log.info(f"🗣️ Lip-synced actor render complete: {out_path}")
    return str(out_path)


def generate_virtual_actor_scene(script_text: str, actor_type: str = "male", mood: str = None):
    """
    Main virtual actor video generator.
    """
    mood = mood or analyze_emotion(script_text)
    voice_path = synthesize_tts(script_text)
    actor_img = generate_actor_image(actor_type, script_text)
    actor_video = simulate_lip_sync(actor_img, voice_path)

    # Apply emotion-based lighting + background music
    final_with_music = attach_music_to_video(actor_video, mood)
    return final_with_music


@app.route("/virtual_actor", methods=["POST"])
def virtual_actor():
    """
    POST JSON:
      {
        "script": "Hello, I am your AI assistant.",
        "actor_type": "female"
      }
    """
    data = request.get_json() or {}
    script = data.get("script", "")
    actor_type = data.get("actor_type", "male")

    if not script:
        return jsonify({"error": "script required"}), 400

    try:
        output = generate_virtual_actor_scene(script, actor_type)
        url = upload_to_firebase(output)
        return jsonify({
            "status": "success",
            "actor_type": actor_type,
            "file": output,
            "cloud_url": url
        })
    except Exception as e:
        log.exception("Virtual actor generation failed")
        return jsonify({"status": "error", "message": str(e)}), 500

# ====================================================
# 🎥 UCVE v8: AI Camera Director (shot planner + director)
# ====================================================
import math
from typing import List, Dict

# Simple planner: create shot intents from script keywords
def plan_camera_shots(script_text: str) -> List[Dict]:
    """
    Return list of shot descriptors:
      { "type": "wide"|"closeup"|"action"|"dramatic",
        "duration": sec, "intensity": 0..1 }
    This is heuristic-based: you can expand rules as needed.
    """
    txt = script_text.lower()
    shots = []

    # Start with opening wide shot
    shots.append({"type": "wide", "duration": 3.0, "intensity": 0.2})

    # If script contains action verbs -> add action shots
    action_keywords = ["run", "chase", "fight", "attack", "explode", "roar", "jump", "fall"]
    if any(k in txt for k in action_keywords):
        shots.append({"type": "action", "duration": 4.0, "intensity": 0.9})

    # If emotional / intimate -> closeup
    intimate_keywords = ["love", "cry", "whisper", "confess", "tears", "heart"]
    if any(k in txt for k in intimate_keywords):
        shots.append({"type": "closeup", "duration": 3.5, "intensity": 0.6})

    # If question/plot-turn -> dramatic
    if "but" in txt or "however" in txt or "suddenly" in txt:
        shots.append({"type": "dramatic", "duration": 3.0, "intensity": 0.7})

    # Ending shot
    shots.append({"type": "wide", "duration": 2.5, "intensity": 0.3})
    return shots


# Director: apply cinematic transforms using MoviePy
def apply_camera_director(video_path: str, shot_plan: List[Dict]) -> str:
    """
    Apply a sequence of cinematic effects to the input video.
    Returns path to final video.
    """
    if not MOVIEPY_AVAILABLE:
        raise RuntimeError("MoviePy not available for camera director")

    from moviepy.editor import VideoFileClip, concatenate_videoclips
    from moviepy.video.fx import all as vfx

    base_clip = VideoFileClip(str(video_path))
    clips = []
    t_cursor = 0.0
    total_duration = base_clip.duration

    # total duration distributed among shots roughly; if mismatch, scale durations
    plan_total = sum(s.get("duration", 3.0) for s in shot_plan) or 1.0
    scale = total_duration / plan_total

    for shot in shot_plan:
        stype = shot.get("type", "wide")
        dur = float(shot.get("duration", 3.0)) * scale
        intensity = float(shot.get("intensity", 0.3))

        # cut a segment from base_clip
        start_t = min(t_cursor, max(0, total_duration - 0.01))
        end_t = min(start_t + dur, total_duration)
        seg = base_clip.subclip(start_t, end_t)

        # apply effects by shot type
        try:
            if stype == "wide":
                # mild color grade + slow zoom-out
                seg = seg.fx(vfx.colorx, 1.02 + 0.01 * intensity)
                seg = seg.resize(lambda t: 1 + 0.005 * math.sin(0.6 * t))
            elif stype == "closeup":
                # gentle zoom-in and slight speed ramp for drama
                seg = seg.resize(lambda t: 1 + 0.015 * (t / max(1.0, dur))
                seg = seg.fx(vfx.lum_contrast, 1.0 + 0.1 * intensity, 1.0, 128)
                seg = seg.fx(vfx.speedx, 1.0 + 0.02 * intensity)
            elif stype == "action":
                # quick zooms + shake (rotate small oscillation)
                seg = seg.fx(vfx.colorx, 1.1)
                seg = seg.resize(lambda t: 1.0 + 0.02 * math.sin(6 * t * intensity))
                seg = seg.fx(vfx.rotate, lambda t: 0.6 * math.sin(12 * t * intensity))
                seg = seg.fx(vfx.speedx, 1.0 + 0.12 * intensity)
            elif stype == "dramatic":
                # slow dolly in + strong contrast
                seg = seg.resize(lambda t: 1 + 0.02 * (t / max(1.0, dur))
                seg = seg.fx(vfx.lum_contrast, 1.2, 1.1, 128)
            else:
                seg = seg.fx(vfx.colorx, 1.0)

            # clamp clip duration to requested dur (some fx may alter length)
            seg = seg.set_duration(end_t - start_t)

        except Exception as e:
            log.warning("Camera director effect failed on segment: %s", e)

        clips.append(seg)
        t_cursor = end_t
        if t_cursor >= total_duration - 0.02:
            break

    # If clips shorter than original (due to rounding), append final tail
    if clips:
        final = concatenate_videoclips(clips, method="compose")
    else:
        final = base_clip

    out_path = OUTPUT_DIR / f"directed_{uuid.uuid4().hex[:8]}.mp4"
    # write out with reasonable settings
    final.write_videofile(str(out_path), fps=24, codec="libx264", audio_codec="aac", threads=2, logger=None)

    try:
        final.close()
        base_clip.close()
    except Exception:
        pass

    log.info("🎬 Camera Director produced -> %s", out_path)
    return str(out_path)


# Flask endpoint to run camera director on a video (or generate UCVE then direct)
@app.route("/camera_direct", methods=["POST"])
def camera_direct_endpoint():
    """
    POST JSON:
     {
       "video_path": "optional local path or 'ucve'",
       "script": "script text used to plan shots (optional if video_path provided)"
     }
    """
    data = request.get_json() or {}
    video_path = data.get("video_path", "")
    script = data.get("script", "")

    if not video_path and not script:
        return jsonify({"error": "Provide video_path or script"}), 400

    try:
        # if video_path requests UCVE generation, create base scene
        if not video_path or video_path == "ucve":
            if "generate_cinematic_scene" in globals():
                base_video = generate_cinematic_scene(script)
            else:
                return jsonify({"error": "UCVE generator not available"}), 400
        else:
            base_video = video_path

        # plan shots (script preferred)
        plan = plan_camera_shots(script if script else " ".join([str(x) for x in []]))
        final_file = apply_camera_director(base_video, plan)
        # optionally upload to firebase if available
        cloud_url = None
        if FIREBASE_BUCKET:
            try:
                cloud_url = upload_to_firebase(final_file)
            except Exception:
                cloud_url = None

        return jsonify({"status": "success", "file": final_file, "cloud_url": cloud_url})
    except Exception as e:
        log.exception("camera_direct endpoint failed: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500

# ====================================================
# 👥 UCVE v9: Real-Time Collaboration + Cloud Edit System
# ====================================================
from datetime import datetime

try:
    from firebase_admin import db as firebase_db
except Exception:
    firebase_db = None

def init_realtime_db():
    """Ensure Firebase Realtime DB initialized."""
    try:
        if not firebase_db:
            return None
        if not firebase_admin._apps:
            init_firebase()
        url = os.getenv("FIREBASE_DB_URL")
        if url:
            firebase_admin.initialize_app(cred, {"databaseURL": url})
        log.info("🔄 Firebase Realtime Database ready.")
    except Exception as e:
        log.warning("Realtime DB init failed: %s", e)

def save_project_state(project_id: str, data: dict):
    """Push project JSON snapshot to Firebase DB."""
    try:
        if not firebase_db:
            raise RuntimeError("Realtime DB not available")
        ref = firebase_db.reference(f"projects/{project_id}")
        data["timestamp"] = datetime.utcnow().isoformat()
        ref.set(data)
        log.info(f"☁️ Project {project_id} synced to cloud")
    except Exception as e:
        log.warning("Project sync failed: %s", e)

def load_project_state(project_id: str) -> dict:
    """Fetch project JSON snapshot from Firebase."""
    if not firebase_db:
        return {}
    try:
        ref = firebase_db.reference(f"projects/{project_id}")
        data = ref.get()
        return data or {}
    except Exception as e:
        log.warning("Fetch project failed: %s", e)
        return {}

@app.route("/collab/create", methods=["POST"])
def collab_create():
    """
    Create a shared project document.
    JSON:
      {"project_id":"demo1","script":"Hero meets tiger","members":["user1","user2"]}
    """
    data = request.get_json() or {}
    project_id = data.get("project_id") or uuid.uuid4().hex[:6]
    script = data.get("script", "")
    members = data.get("members", [])
    state = {
        "project_id": project_id,
        "script": script,
        "members": members,
        "created_at": datetime.utcnow().isoformat(),
        "edits": []
    }
    save_project_state(project_id, state)
    return jsonify({"status":"success","project_id":project_id})

@app.route("/collab/edit", methods=["POST"])
def collab_edit():
    """
    Append edit/comment to project.
    JSON:
      {"project_id":"demo1","user":"user1","edit":"Changed tiger roar to whisper"}
    """
    data = request.get_json() or {}
    pid = data.get("project_id")
    user = data.get("user","anon")
    edit = data.get("edit","")
    if not pid or not edit:
        return jsonify({"error":"project_id and edit required"}),400
    state = load_project_state(pid)
    edits = state.get("edits", [])
    edits.append({"user":user,"edit":edit,"time":datetime.utcnow().isoformat()})
    state["edits"] = edits
    save_project_state(pid, state)
    return jsonify({"status":"ok","project_id":pid,"total_edits":len(edits)})

@app.route("/collab/fetch", methods=["GET"])
def collab_fetch():
    """
    GET ?project_id=demo1
    """
    pid = request.args.get("project_id")
    if not pid:
        return jsonify({"error":"project_id required"}),400
    data = load_project_state(pid)
    return jsonify(data)

# simple AI merge stub (placeholder for GPT suggestions)
def merge_script_edits(base_script: str, edits: list) -> str:
    merged = base_script
    for e in edits:
        merged += f"\n# {e['user']} suggestion: {e['edit']}"
    return merged

@app.route("/collab/merge", methods=["POST"])
def collab_merge():
    """
    Merge all edits into one script (AI-assisted stub).
    JSON: {"project_id":"demo1"}
    """
    data = request.get_json() or {}
    pid = data.get("project_id")
    state = load_project_state(pid)
    merged_script = merge_script_edits(state.get("script",""), state.get("edits",[]))
    return jsonify({"status":"merged","merged_script":merged_script})

# ====================================================
# 💼 UCVE v10: AI Project Manager Dashboard & Analytics
# ====================================================
import statistics
from collections import Counter

def compute_dashboard_metrics():
    """
    Calculate total users, videos, credits, and job success stats.
    """
    users = UserProfile.query.count()
    videos = UserVideo.query.count()
    success_jobs = UserVideo.query.filter_by(status="done").count()
    failed_jobs = UserVideo.query.filter_by(status="failed").count()
    total_credits = sum(u.credits for u in UserProfile.query.all())

    # derive simple ratios
    success_rate = round((success_jobs / videos) * 100, 2) if videos > 0 else 0.0
    fail_rate = 100 - success_rate

    return {
        "total_users": users,
        "total_videos": videos,
        "success_jobs": success_jobs,
        "failed_jobs": failed_jobs,
        "success_rate": success_rate,
        "fail_rate": fail_rate,
        "total_credits": total_credits
    }

def get_trending_templates():
    """
    Detect trending template categories by usage frequency.
    """
    tdata = TemplateCatalog.query.order_by(TemplateCatalog.trending_score.desc()).limit(10).all()
    return [{"name": t.name, "category": t.category, "score": t.trending_score} for t in tdata]

def get_active_users(limit: int = 5):
    """
    Top active users (by videos rendered).
    """
    counts = Counter([v.user_email for v in UserVideo.query.all()])
    top_users = counts.most_common(limit)
    out = []
    for email, cnt in top_users:
        u = UserProfile.query.filter_by(email=email).first()
        if u:
            out.append({
                "email": u.email,
                "name": u.name,
                "videos_rendered": cnt,
                "credits_left": u.credits,
                "plan": u.plan
            })
    return out

def analyze_system_health():
    """
    Create a simple health summary.
    """
    data = compute_dashboard_metrics()
    msg = []
    if data["success_rate"] < 60:
        msg.append("⚠️ Rendering success rate below 60%")
    else:
        msg.append("✅ Rendering pipeline healthy")

    if data["total_credits"] < 50:
        msg.append("⚠️ Low global credits; top-up needed")

    if data["total_users"] > 1000:
        msg.append("🔥 High user growth detected")

    return " | ".join(msg)

def generate_ai_insights():
    """
    Analyze video titles and scripts for trending topics using TextBlob sentiment.
    """
    vids = UserVideo.query.order_by(UserVideo.created_at.desc()).limit(50).all()
    categories = []
    sentiments = []
    for v in vids:
        try:
            txt = (v.title or "") + " " + (v.script or "")
            blob = textblob.TextBlob(txt)
            sentiments.append(blob.sentiment.polarity)
            categories.append(v.template or "General")
        except Exception:
            pass

    avg_sent = statistics.mean(sentiments) if sentiments else 0.0
    trend = Counter(categories).most_common(3)
    return {
        "avg_sentiment": avg_sent,
        "top_categories": [t[0] for t in trend],
        "message": "Viewers reacting positively!" if avg_sent > 0 else "Content trend: neutral or mixed."
    }

@app.route("/dashboard/summary", methods=["GET"])
def dashboard_summary():
    data = compute_dashboard_metrics()
    data["health"] = analyze_system_health()
    data["top_users"] = get_active_users()
    return jsonify(data)

@app.route("/dashboard/trending", methods=["GET"])
def dashboard_trending():
    t = get_trending_templates()
    return jsonify({"trending": t})

@app.route("/dashboard/insights", methods=["GET"])
def dashboard_insights():
    info = generate_ai_insights()
    return jsonify(info)

# ====================================================
# 🔐 Payments Integration: Stripe Checkout + Razorpay Orders + Webhooks
# ====================================================
import os
import stripe
import hmac
import hashlib
import time
from flask import request, jsonify, abort

# --- Read env keys ---
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET")
DOMAIN = os.getenv("DOMAIN", "http://127.0.0.1:5000")  # change to https://yourdomain.com in prod

# --- Init stripe if key available ---
STRIPE_READY = False
if STRIPE_SECRET_KEY:
    try:
        stripe.api_key = STRIPE_SECRET_KEY
        STRIPE_READY = True
        log.info("Stripe configured.")
    except Exception as e:
        log.warning("Stripe init failed: %s", e)

# --- Razorpay simple availability flag (use requests fallback if SDK not installed) ---
RAZORPAY_READY = False
try:
    import razorpay
    if RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET:
        razorpay_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))
        RAZORPAY_READY = True
        log.info("Razorpay configured.")
except Exception as e:
    log.info("Razorpay SDK not available: %s", e)
    RAZORPAY_READY = False

# ---------- Stripe: create checkout session ----------
@app.route("/stripe/create_checkout", methods=["POST"])
def stripe_create_checkout():
    """
    POST JSON:
      { "price_cents": 49900, "currency": "inr", "user_email": "demo@visora.com", "metadata": {...} }
    Returns:
      { "sessionId": "<id>", "url": "<checkout_url>" }
    """
    if not STRIPE_READY:
        return jsonify({"error": "Stripe not configured on server"}), 501

    data = request.get_json() or {}
    price_cents = int(data.get("price_cents", 100))
    currency = data.get("currency", "inr")
    user_email = data.get("user_email", "demo@visora.com")
    metadata = data.get("metadata", {})

    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            mode="payment",
            line_items=[{
                "price_data": {
                    "currency": currency,
                    "product_data": {"name": metadata.get("product_name", "Visora Credits")},
                    "unit_amount": price_cents
                },
                "quantity": 1
            }],
            customer_email=user_email,
            metadata=metadata,
            success_url=f"{DOMAIN}/payments/success?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{DOMAIN}/payments/cancel"
        )
        return jsonify({"sessionId": session.id, "url": session.url})
    except Exception as e:
        log.exception("Stripe checkout create failed")
        return jsonify({"error": str(e)}), 500

# ---------- Stripe webhook handler ----------
@app.route("/stripe/webhook", methods=["POST"])
def stripe_webhook():
    payload = request.data
    sig_header = request.headers.get("Stripe-Signature", "")
    event = None

    # Verify webhook signature if secret provided
    if STRIPE_WEBHOOK_SECRET:
        try:
            event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
        except Exception as e:
            log.exception("Stripe webhook signature verification failed")
            return jsonify({"error": "invalid signature"}), 400
    else:
        try:
            event = request.get_json()
        except Exception:
            return jsonify({"error": "invalid payload"}), 400

    # Handle events
    etype = event.get("type") if isinstance(event, dict) else getattr(event, "type", None)
    log.info("Stripe webhook event: %s", etype)

    # Payment succeeded
    if etype == "checkout.session.completed" or etype == "payment_intent.succeeded":
        # retrieve relevant info
        obj = event.get("data", {}).get("object", {}) if isinstance(event, dict) else event.data.object
        metadata = obj.get("metadata", {}) if isinstance(obj, dict) else {}
        user_email = obj.get("customer_email") or metadata.get("user_email") or metadata.get("email")
        amount = obj.get("amount_total") or obj.get("amount")  # cents
        # allocate credits or upgrade plan logic here
        try:
            if user_email:
                u = UserProfile.query.filter_by(email=user_email).first()
                if u:
                    # example: 1 credit = ₹1 -> allocate amount/100 credits
                    credits_to_add = int((amount or 0) // 100)
                    u.credits = (u.credits or 0) + credits_to_add
                    db.session.commit()
                    log.info("Allocated %s credits to %s", credits_to_add, user_email)
        except Exception:
            log.exception("Failed to allocate credits after stripe webhook")

    return jsonify({"status": "ok"}), 200

# ---------- Razorpay: create order ----------
@app.route("/razorpay/create_order", methods=["POST"])
def razorpay_create_order():
    """
    POST JSON:
      { "amount": 49900, "currency": "INR", "receipt": "rcpt_123", "notes": {...} }
    Returns order object for client-side checkout
    """
    data = request.get_json() or {}
    amount = int(data.get("amount", 0))
    currency = data.get("currency", "INR")
    receipt = data.get("receipt", f"rcpt_{int(time.time())}")
    notes = data.get("notes", {})

    if not RAZORPAY_READY:
        return jsonify({"error": "Razorpay not configured on server"}), 501

    try:
        order = razorpay_client.order.create({"amount": amount, "currency": currency, "receipt": receipt, "notes": notes})
        return jsonify({"order": order})
    except Exception as e:
        log.exception("Razorpay order creation failed")
        return jsonify({"error": str(e)}), 500

# ---------- Razorpay: verify payment signature ----------
@app.route("/razorpay/verify", methods=["POST"])
def razorpay_verify():
    """
    POST JSON:
      { "razorpay_order_id": "...", "razorpay_payment_id":"...", "razorpay_signature":"...", "user_email":"..." }
    """
    data = request.get_json() or {}
    order_id = data.get("razorpay_order_id")
    payment_id = data.get("razorpay_payment_id")
    signature = data.get("razorpay_signature")
    user_email = data.get("user_email")

    if not (order_id and payment_id and signature):
        return jsonify({"error": "missing params"}), 400

    # verify signature
    try:
        if RAZORPAY_READY:
            msg = "{}|{}".format(order_id, payment_id)
            generated_signature = hmac.new(RAZORPAY_KEY_SECRET.encode(), msg.encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(generated_signature, signature):
                return jsonify({"error": "invalid signature"}), 400
            # success -> allocate credits (example)
            try:
                if user_email:
                    u = UserProfile.query.filter_by(email=user_email).first()
                    if u:
                        # amount fetch from order API if needed
                        # for demo, add fixed credits
                        u.credits = (u.credits or 0) + 50
                        db.session.commit()
                return jsonify({"status": "ok"})
            except Exception:
                log.exception("Allocate after razorpay verify failed")
                return jsonify({"status":"error"}), 500
        else:
            return jsonify({"error":"Razorpay not configured"}), 501
    except Exception as e:
        log.exception("Razorpay verify failed: %s", e)
        return jsonify({"error": str(e)}), 500

# ---------- Simple client checkout HTML for Stripe (return URL) ----------
@app.route("/payments/stripe_checkout_page", methods=["GET"])
def stripe_checkout_page():
    # small static HTML that demonstrates usage of sessionId returned by /stripe/create_checkout
    html = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <title>Stripe Checkout Demo</title>
      <script src="https://js.stripe.com/v3/"></script>
    </head>
    <body>
      <h3>Stripe Checkout Demo</h3>
      <p>Open browser console for errors.</p>
      <button id="checkout">Pay</button>
      <script>
        async function start() {{
          // create session on server
          const res = await fetch('/stripe/create_checkout', {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json' }},
            body: JSON.stringify({{ price_cents: 49900, currency: 'inr', user_email: 'demo@visora.com', metadata: {{product_name:'Visora Premium'}} }})
          }});
          const data = await res.json();
          if(data.error) {{
            alert("Error: " + data.error);
            return;
          }}
          const stripe = Stripe('{stripe_publishable_placeholder}');
          stripe.redirectToCheckout({{ sessionId: data.sessionId }});
        }}

        document.getElementById('checkout').addEventListener('click', start);
      </script>
    </body>
    </html>
    """
    return html

# ---------- Notes endpoint ----------
@app.route("/payments/notes", methods=["GET"])
def payments_notes():
    return jsonify({
        "stripe_ready": STRIPE_READY,
        "razorpay_ready": RAZORPAY_READY,
        "domain": DOMAIN,
        "note": "Set STRIPE_SECRET_KEY, STRIPE_WEBHOOK_SECRET, RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET in env"
    })

# ====================================================
# 🌌 UCVE v13: 3D Scene Composer (Virtual Camera, Lights, Motion Path)
# ====================================================
import numpy as np
import cv2
from moviepy.editor import *
import open3d as o3d

def generate_3d_background(theme: str = "sunset"):
    """
    Generate a pseudo-3D background using Open3D.
    Returns path to rendered background image (PNG).
    """
    w, h = 1280, 720
    img = np.zeros((h, w, 3), dtype=np.uint8)

    if theme == "sunset":
        img[:, :] = (20, 50, 100)
        cv2.circle(img, (w // 2, h // 2), 180, (255, 160, 90), -1)
    elif theme == "forest":
        img[:, :] = (30, 80, 30)
        cv2.rectangle(img, (0, h//2), (w, h), (40, 120, 40), -1)
    elif theme == "city":
        img[:, :] = (30, 30, 30)
        for i in range(0, w, 80):
            cv2.rectangle(img, (i, h//2), (i+40, h), (80, 80, 100), -1)
    else:
        img[:, :] = (50, 60, 70)

    out_path = str(OUTPUT_FOLDER / f"background_{theme}.png")
    cv2.imwrite(out_path, img)
    return out_path


def apply_virtual_lighting(image_path: str, mood: str = "dramatic"):
    """
    Apply lighting and depth to 2D image.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Invalid image path")

    overlay = img.copy()
    h, w = img.shape[:2]
    mask = np.zeros((h, w, 3), dtype=np.uint8)

    if mood == "dramatic":
        cv2.circle(mask, (w//2, h//2), w//2, (60, 60, 60), -1)
        cv2.addWeighted(mask, 0.5, img, 0.7, 0, overlay)
    elif mood == "happy":
        cv2.rectangle(mask, (0, 0), (w, h), (255, 255, 100), -1)
        cv2.addWeighted(mask, 0.3, img, 0.8, 0, overlay)
    else:
        cv2.GaussianBlur(img, (9,9), 0)

    out_path = str(OUTPUT_FOLDER / f"lit_{uuid.uuid4().hex[:8]}.png")
    cv2.imwrite(out_path, overlay)
    return out_path


def create_motion_path_scene(bg_path: str, char_img_path: str, motion: str = "pan-left"):
    """
    Create a 3D motion path render by animating camera movement.
    """
    if not MOVIEPY_AVAILABLE:
        raise RuntimeError("MoviePy not available for motion rendering")

    bg_clip = ImageClip(bg_path).set_duration(6)
    char_clip = ImageClip(char_img_path).resize(height=400).set_position(("center", "center")).set_duration(6)

    if motion == "pan-left":
        char_clip = char_clip.set_position(lambda t: ("center", 360 - 50 * t))
    elif motion == "zoom-in":
        char_clip = char_clip.resize(lambda t: 1 + 0.05 * t)
    elif motion == "orbit":
        char_clip = char_clip.set_position(lambda t: (640 + 80 * np.sin(t * 2), 360 + 50 * np.cos(t * 2))

    final = CompositeVideoClip([bg_clip, char_clip])
    out_path = str(OUTPUT_FOLDER / f"scene3d_{uuid.uuid4().hex[:8]}.mp4")
    final.write_videofile(out_path, fps=24, codec="libx264", audio_codec="aac")
    return out_path


@app.route("/ucve_3d_scene", methods=["POST"])
def ucve_3d_scene():
    """
    POST JSON:
      {
        "theme": "forest",
        "mood": "dramatic",
        "motion": "orbit",
        "character_image": "uploads/characters/abc.png"
      }
    """
    data = request.get_json() or {}
    theme = data.get("theme", "sunset")
    mood = data.get("mood", "dramatic")
    motion = data.get("motion", "pan-left")
    char_path = data.get("character_image")

    try:
        bg_path = generate_3d_background(theme)
        lit_char = apply_virtual_lighting(_abs_path(char_path), mood)
        final_video = create_motion_path_scene(bg_path, lit_char, motion)

        cloud_url = None
        if FIREBASE_BUCKET:
            cloud_url = upload_to_firebase(final_video)

        return jsonify({
            "status": "success",
            "theme": theme,
            "mood": mood,
            "motion": motion,
            "video": final_video,
            "cloud_url": cloud_url
        })
    except Exception as e:
        log.exception("3D scene generation failed")
        return jsonify({"status": "error", "message": str(e)}), 500

# ====================================================
# 🎯 UCVE v14: AI Camera Path Optimizer (Smooth Bezier Camera Paths)
# ====================================================
import math
from typing import List, Tuple

# Helpers: cubic bezier interpolation for 2D points
def cubic_bezier(p0, p1, p2, p3, t: float) -> Tuple[float, float]:
    """Return point on cubic bezier at parameter t in [0,1]."""
    u = 1 - t
    b0 = (u**3)
    b1 = 3 * (u**2) * t
    b2 = 3 * u * (t**2)
    b3 = t**3
    x = b0 * p0[0] + b1 * p1[0] + b2 * p2[0] + b3 * p3[0]
    y = b0 * p0[1] + b1 * p1[1] + b2 * p2[1] + b3 * p3[1]
    return (x, y)

def linear_interp(a, b, t: float):
    return a + (b - a) * t

def plan_camera_path_from_script(script_text: str, width:int=1280, height:int=720) -> List[dict]:
    """
    Heuristic planner: produce list of keyframes.
    Each keyframe: {"time": seconds, "cx": center_x (0..1), "cy": center_y (0..1), "zoom": zoom_factor}
    - cx/cy normalized to 0..1 (relative to frame)
    - zoom: 1.0 = full frame, >1 = zoom-in (crop)
    """
    txt = (script_text or "").lower()
    base_duration = 6.0  # default shot length (will be rescaled to actual clip length)

    keyframes = []
    # start: wide center
    keyframes.append({"time":0.0,"cx":0.5,"cy":0.5,"zoom":1.0})

    # if action words -> quick push-in to left or right
    if any(k in txt for k in ["run","chase","attack","roar","explode","fight"]):
        keyframes.append({"time": base_duration*0.25, "cx":0.65, "cy":0.45, "zoom":1.15})
        keyframes.append({"time": base_duration*0.6, "cx":0.4, "cy":0.5, "zoom":1.25})
    # if intimate dialog -> slow dolly-in center
    if any(k in txt for k in ["love","cry","whisper","confess","tears","heart"]):
        keyframes.append({"time": base_duration*0.35, "cx":0.5, "cy":0.45, "zoom":1.18})
        keyframes.append({"time": base_duration*0.85, "cx":0.5, "cy":0.42, "zoom":1.35})
    # if sudden/turn -> dramatic pan
    if any(k in txt for k in ["suddenly","but","however","then"]):
        keyframes.append({"time": base_duration*0.5, "cx":0.2, "cy":0.55, "zoom":1.12})

    # always end with a calm wide shot
    keyframes.append({"time": base_duration, "cx":0.5, "cy":0.5, "zoom":1.0})

    # normalize times to 0..1 range for interpolation stage (actual durations handled later)
    return keyframes

def expand_keyframes_to_bezier_segments(keyframes: List[dict]) -> List[dict]:
    """
    Convert simple keyframes into segments with bezier control points.
    For each consecutive 4 control points, we'll interpret as cubic bezier.
    We'll produce a list of segments each with p0..p3 points in normalized coords.
    """
    pts = [(kf["cx"], kf["cy"], kf["zoom"], kf["time"]) for kf in keyframes]
    # convert into list of 2D positions and zooms
    pos = [(p[0], p[1]) for p in pts]
    zooms = [p[2] for p in pts]
    times = [p[3] for p in pts]

    segments = []
    n = max(1, len(pos)-1)
    # create simple control points: for p_i create p_i, p_i + delta/3, p_{i+1} - delta/3, p_{i+1}
    for i in range(n):
        p0 = pos[i]
        p3 = pos[min(i+1, len(pos)-1)]
        # delta
        dx = (p3[0]-p0[0])
        dy = (p3[1]-p0[1])
        p1 = (p0[0] + dx*0.33, p0[1] + dy*0.33)
        p2 = (p0[0] + dx*0.66, p0[1] + dy*0.66)
        t0 = times[i]
        t1 = times[min(i+1, len(times)-1)]
        z0 = zooms[i]
        z1 = zooms[min(i+1, len(zooms)-1)]
        segments.append({
            "p0": p0, "p1": p1, "p2": p2, "p3": p3,
            "t0": t0, "t1": t1,
            "z0": z0, "z1": z1
        })
    return segments

def apply_camera_path(video_path: str, keyframes: List[dict], out_name: str = None) -> str:
    """
    Apply camera path to the video by using MoviePy's crop with time-varying x1/y1/x2/y2 functions.
    keyframes: list produced by plan_camera_path_from_script
    Returns path to final video.
    """
    if not MOVIEPY_AVAILABLE:
        raise RuntimeError("MoviePy required for camera optimizer")

    from moviepy.editor import VideoFileClip
    from moviepy.video.fx import crop

    clip = VideoFileClip(str(video_path))
    w, h = clip.w, clip.h
    duration = clip.duration or 6.0

    # Normalize keyframe times to actual clip duration
    kfs = []
    last_time = keyframes[-1]["time"] if keyframes else 1.0
    if last_time <= 0:
        last_time = 1.0
    for k in keyframes:
        kfs.append({
            "time": (k["time"] / last_time) * duration,
            "cx": k["cx"],
            "cy": k["cy"],
            "zoom": k["zoom"]
        })

    # Expand to segments with bezier control points
    segs = expand_keyframes_to_bezier_segments(kfs)

    # For fast lookup choose function that maps t -> (cx,cy,zoom)
    def lookup_at(t):
        # find segment containing t (by t0..t1)
        if not segs:
            return (0.5,0.5,1.0)
        for s in segs:
            if t >= s["t0"] and t <= s["t1"]:
                # map t to local u
                span = max(1e-6, s["t1"] - s["t0"])
                u = (t - s["t0"]) / span
                cx, cy = cubic_bezier(s["p0"], s["p1"], s["p2"], s["p3"], u)
                zoom = linear_interp(s["z0"], s["z1"], u)
                return (cx, cy, zoom)
        # fallback last
        last = segs[-1]
        return (last["p3"][0], last["p3"][1], last["z1"])

    # Define time-varying crop bounds (x1,y1,x2,y2)
    def x1_at(t):
        cx, cy, zoom = lookup_at(t)
        crop_w = w / zoom
        x1 = max(0, int(cx * w - crop_w/2))
        return x1

    def y1_at(t):
        cx, cy, zoom = lookup_at(t)
        crop_h = h / zoom
        y1 = max(0, int(cy * h - crop_h/2))
        return y1

    def x2_at(t):
        cx, cy, zoom = lookup_at(t)
        crop_w = w / zoom
        x2 = min(w, int(cx * w + crop_w/2))
        return x2

    def y2_at(t):
        cx, cy, zoom = lookup_at(t)
        crop_h = h / zoom
        y2 = min(h, int(cy * h + crop_h/2))
        return y2

    # Apply crop with functions
    try:
        cropped = clip.fx(crop, x1=lambda t: x1_at(t), y1=lambda t: y1_at(t),
                          x2=lambda t: x2_at(t), y2=lambda t: y2_at(t))
        out_path = OUTPUT_DIR / f"camopt_{uuid.uuid4().hex[:8]}.mp4" if not out_name else OUTPUT_DIR / out_name
        cropped.write_videofile(str(out_path), fps=24, codec="libx264", audio_codec="aac", logger=None)
        try:
            cropped.close()
            clip.close()
        except Exception:
            pass
        return str(out_path)
    except Exception as e:
        log.exception("Camera path application failed: %s", e)
        clip.close()
        raise

@app.route("/camera_optimize", methods=["POST"])
def camera_optimize_endpoint():
    """
    POST JSON:
      { "video_path": "path/to/video.mp4" OR "ucve", "script":"...", "fov_zoom":1.15 }
    Response:
      { "status":"success", "file": "<path>", "cloud_url": "<optional>" }
    """
    data = request.get_json() or {}
    video_path = data.get("video_path", "")
    script = data.get("script", "")
    fov_zoom = float(data.get("fov_zoom", 1.0))

    if not video_path and not script:
        return jsonify({"error":"provide video_path or script"}), 400

    try:
        # generate base if requested
        if video_path == "ucve" or not video_path:
            if "generate_cinematic_scene" in globals():
                base_video = generate_cinematic_scene(script)
            else:
                return jsonify({"error":"no base video available"}), 400
        else:
            base_video = video_path

        # plan keyframes
        keyframes = plan_camera_path_from_script(script)
        # optionally apply global fov_zoom scaling
        for k in keyframes:
            k["zoom"] = k.get("zoom",1.0) * fov_zoom

        final_file = apply_camera_path(base_video, keyframes)
        cloud_url = None
        if FIREBASE_BUCKET:
            try:
                cloud_url = upload_to_firebase(final_file)
            except Exception:
                cloud_url = None

        return jsonify({"status":"success","file": final_file, "cloud_url": cloud_url})
    except Exception as e:
        log.exception("camera_optimize failed")
        return jsonify({"status":"error","message": str(e)}), 500

# ====================================================
# 🌌 UCVE v15: Real Depth Parallax Engine (AI Depth Map Simulation)
# ====================================================
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2

# --- Depth Model Cache ---
_midas_model = None
_midas_transform = None

def load_midas_model():
    """Load MiDaS small model once (for CPU)."""
    global _midas_model, _midas_transform
    if _midas_model is not None:
        return _midas_model, _midas_transform
    try:
        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        midas.eval()
        transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
        _midas_model, _midas_transform = midas, transform
        log.info("MiDaS depth model loaded ✅")
        return midas, transform
    except Exception as e:
        log.warning("MiDaS model load failed: %s", e)
        return None, None

def generate_depth_map(image_path: str, out_path: Optional[str] = None) -> Optional[str]:
    """Generate AI depth map for an image (0–255 grayscale)."""
    model, transform = load_midas_model()
    if model is None:
        raise RuntimeError("MiDaS not available")

    img = Image.open(image_path).convert("RGB")
    inp = transform(img).unsqueeze(0)
    with torch.no_grad():
        pred = model(inp)
        depth = pred.squeeze().cpu().numpy()

    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    out = out_path or str(OUTPUT_FOLDER / f"depth_{uuid.uuid4().hex[:8]}.png")
    cv2.imwrite(out, depth_norm)
    return out


def apply_parallax_motion(image_path: str, depth_path: str, motion: str = "dolly-in", duration: float = 6.0) -> str:
    """
    Combine image + depth to generate 3D-like camera movement.
    """
    img = cv2.imread(image_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    if img is None or depth is None:
        raise ValueError("Missing image/depth input")

    h, w = img.shape[:2]
    frames = []
    n_frames = int(duration * 24)
    for i in range(n_frames):
        t = i / n_frames
        if motion == "dolly-in":
            scale = 1.0 + 0.05 * t
            dx = int((depth * (t * 5 / 255)).mean())
            M = np.float32([[scale, 0, -dx], [0, scale, 0]])
        elif motion == "pan-left":
            dx = int(20 * t)
            M = np.float32([[1, 0, -dx], [0, 1, 0]])
        elif motion == "tilt-up":
            dy = int(15 * t)
            M = np.float32([[1, 0, 0], [0, 1, -dy]])
        else:
            M = np.eye(2, 3, dtype=np.float32)

        transformed = cv2.warpAffine(img, M, (w, h))
        blur_amt = int(4 + 8 * (1 - t))
        depth_blur = cv2.GaussianBlur(transformed, (blur_amt | 1, blur_amt | 1), 0)
        frames.append(depth_blur)

    out_path = str(OUTPUT_FOLDER / f"parallax_{uuid.uuid4().hex[:8]}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, 24, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    return out_path


@app.route("/ucve_depth_parallax", methods=["POST"])
def ucve_depth_parallax():
    """
    POST JSON:
      {
        "image_path": "uploads/characters/demo.png",
        "motion": "dolly-in",
        "duration": 6.0
      }
    Returns:
      { "status":"success", "depth_map":"...", "video":"...", "cloud_url":"..." }
    """
    data = request.get_json() or {}
    img = data.get("image_path")
    motion = data.get("motion", "dolly-in")
    duration = float(data.get("duration", 6.0))

    try:
        depth_path = generate_depth_map(_abs_path(img))
        video_path = apply_parallax_motion(_abs_path(img), depth_path, motion, duration)

        cloud_url = None
        if FIREBASE_BUCKET:
            try:
                cloud_url = upload_to_firebase(video_path)
            except Exception:
                cloud_url = None

        return jsonify({
            "status":"success",
            "depth_map": depth_path,
            "video": video_path,
            "cloud_url": cloud_url
        })
    except Exception as e:
        log.exception("Depth parallax failed")
        return jsonify({"status":"error", "message": str(e)}), 500

# ====================================================
# 🎨 UCVE v16: AI Cinematic Color Grading & Mood Filters
# ====================================================
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip, vfx

# Pre-defined simple LUT / grading functions
def apply_teal_orange(image: np.ndarray) -> np.ndarray:
    """Simple teal-orange boost: shift shadows to teal, highlights to warm."""
    img = image.astype(np.float32) / 255.0
    # split channels
    r, g, b = img[...,2], img[...,1], img[...,0]
    # boost warm in highlights (increase red), teal in shadows (increase blue/green)
    lum = 0.299*r + 0.587*g + 0.114*b
    highlights = np.clip((lum - 0.5) * 2.0, 0,1)
    shadows = 1.0 - highlights
    r = np.clip(r + highlights * 0.12, 0,1)
    g = np.clip(g + shadows * 0.04, 0,1)
    b = np.clip(b + shadows * 0.08 - highlights * 0.02, 0,1)
    out = np.stack([b,g,r], axis=-1)
    out = np.clip(out * 255.0, 0,255).astype(np.uint8)
    return out

def apply_warm_filter(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32)
    lut = np.arange(0,256).astype(np.uint8)
    # slight gamma warm
    gamma = 0.95
    img = 255.0 * ((img/255.0) ** gamma)
    # boost reds
    img[...,2] = np.clip(img[...,2] * 1.06, 0,255)
    return img.astype(np.uint8)

def apply_noir_filter(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # high contrast
    eq = cv2.equalizeHist(gray)
    out = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
    return out

def apply_cold_filter(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32) / 255.0
    r, g, b = img[...,2], img[...,1], img[...,0]
    r = np.clip(r * 0.95, 0,1)
    b = np.clip(b * 1.08, 0,1)
    out = np.stack([b,g,r], axis=-1)
    out = np.clip(out * 255.0, 0,255).astype(np.uint8)
    return out

# Tone mapping / contrast utility
def contrast_and_vignette(image: np.ndarray, contrast: float = 1.05, vignette_strength: float = 0.4) -> np.ndarray:
    img = image.astype(np.float32)
    # contrast
    mean = np.mean(img, axis=(0,1), keepdims=True)
    img = (img - mean) * contrast + mean
    img = np.clip(img, 0,255).astype(np.uint8)
    # vignette
    h, w = img.shape[:2]
    X = np.linspace(-1,1,w)
    Y = np.linspace(-1,1,h)
    xv, yv = np.meshgrid(X, Y)
    mask = 1.0 - vignette_strength * ((xv**2 + yv**2))
    mask = np.clip(mask, 0.0, 1.0)[:,:,None]
    img = (img.astype(np.float32) * mask).astype(np.uint8)
    return img

# High-level apply mood
def grade_frame_by_mood(frame: np.ndarray, mood: str) -> np.ndarray:
    if mood == "happy":
        out = apply_teal_orange(frame)
        out = contrast_and_vignette(out, contrast=1.03, vignette_strength=0.18)
    elif mood == "dramatic" or mood == "sad":
        out = apply_cold_filter(frame)
        out = contrast_and_vignette(out, contrast=1.08, vignette_strength=0.35)
    elif mood == "noir":
        out = apply_noir_filter(frame)
        out = contrast_and_vignette(out, contrast=1.12, vignette_strength=0.45)
    elif mood == "warm":
        out = apply_warm_filter(frame)
        out = contrast_and_vignette(out, contrast=1.02, vignette_strength=0.12)
    else:
        out = contrast_and_vignette(frame, contrast=1.02, vignette_strength=0.12)
    return out

# MoviePy-friendly filter (applies per-frame)
def color_grade_video(input_video: str, mood: str = "dramatic", out_name: str = None) -> str:
    if not MOVIEPY_AVAILABLE:
        raise RuntimeError("moviepy required for color grading")

    clip = VideoFileClip(str(input_video))
    # apply frame-by-frame transform (fast enough for short clips)
    def fl_image(get_frame, t):
        frame = get_frame(t)
        # frame is RGB in MoviePy; convert to BGR for OpenCV operations
        frame_bgr = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        graded = grade_frame_by_mood(frame_bgr, mood)
        graded_rgb = cv2.cvtColor(graded, cv2.COLOR_BGR2RGB)
        return (graded_rgb.astype(np.uint8) / 255.0)

    graded = clip.fl_image(lambda frame: fl_image(lambda t: frame, 0))
    # NOTE: .fl_image expects a function receiving frame array directly; above lambda wraps
    # Save output
    if not out_name:
        out_path = OUTPUT_DIR / f"graded_{uuid.uuid4().hex[:8]}.mp4"
    else:
        out_path = OUTPUT_DIR / out_name
    graded.write_videofile(str(out_path), fps=clip.fps or 24, codec="libx264", audio_codec="aac", threads=2, logger=None)
    try:
        graded.close()
        clip.close()
    except Exception:
        pass
    return str(out_path)

# Flask endpoint: grade by mood
@app.route("/color_grade", methods=["POST"])
def color_grade_endpoint():
    """
    POST JSON:
      {
        "video_path": "outputs/some.mp4" or "ucve",
        "mood": "dramatic" | "happy" | "warm" | "noir" | "cold"
      }
    """
    data = request.get_json() or {}
    video_path = data.get("video_path", "")
    mood = data.get("mood", "dramatic")
    if not video_path:
        return jsonify({"error":"video_path required"}), 400

    try:
        if video_path == "ucve" and "generate_cinematic_scene" in globals():
            video_path = generate_cinematic_scene(data.get("script",""))
        out = color_grade_video(video_path, mood=mood)
        cloud_url = None
        if FIREBASE_BUCKET:
            try:
                cloud_url = upload_to_firebase(out)
            except Exception:
                cloud_url = None
        return jsonify({"status":"success","file": out,"cloud_url": cloud_url})
    except Exception as e:
        log.exception("Color grading failed: %s", e)
        return jsonify({"status":"error","message": str(e)}), 500

# ====================================================
# ⚙️ UCVE v17: Production Hardening, Scaling & Observability
# - Redis + RQ background queue (preferred)
# - Thread fallback queue (if Redis not available)
# - Prometheus metrics endpoint
# - Graceful shutdown handling
# ====================================================
import atexit
import signal
import threading
import time
import os
from prometheus_client import Counter, Gauge, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST

# --------- Redis + RQ (optional) ----------
REDIS_URL = os.getenv("REDIS_URL", None)
USE_RQ = False
try:
    if REDIS_URL:
        import redis
        from rq import Queue, Connection, Worker
        redis_conn = redis.from_url(REDIS_URL)
        rq_queue = Queue("visora_jobs", connection=redis_conn, default_timeout=60*60*2)
        USE_RQ = True
        log.info("RQ queue initialized (Redis): %s", REDIS_URL)
    else:
        USE_RQ = False
        log.info("REDIS_URL not set — RQ disabled, using thread fallback.")
except Exception as e:
    USE_RQ = False
    log.warning("RQ/Redis not available: %s", e)

# --------- Thread fallback worker ----------
thread_worker_running = False
thread_queue = []
thread_lock = threading.Lock()

def thread_worker_loop():
    global thread_worker_running
    thread_worker_running = True
    log.info("Thread fallback worker started")
    while thread_worker_running:
        job = None
        with thread_lock:
            if thread_queue:
                job = thread_queue.pop(0)
        if job:
            try:
                job_id = job.get("job_id")
                job_fn = job.get("fn")
                job_args = job.get("args", ())
                job_kwargs = job.get("kwargs", {})
                # run the job
                render_jobs[job_id]["status"] = "running"
                job_fn(*job_args, **job_kwargs)
                render_jobs[job_id]["status"] = "done"
            except Exception as e:
                log.exception("Thread worker job failed: %s", e)
                render_jobs[job_id]["status"] = "failed"
        else:
            time.sleep(0.5)

def enqueue_thread_job(job):
    with thread_lock:
        thread_queue.append(job)

# start thread worker as daemon
thread_worker = threading.Thread(target=thread_worker_loop, daemon=True)
thread_worker.start()

# --------- Metrics (Prometheus) ----------
REGISTRY = CollectorRegistry()
MET_JOB_SUBMITTED = Counter('visora_jobs_submitted_total', 'Total render jobs submitted', registry=REGISTRY)
MET_JOB_DONE = Counter('visora_jobs_done_total', 'Total render jobs completed', registry=REGISTRY)
MET_JOB_FAILED = Counter('visora_jobs_failed_total', 'Total render jobs failed', registry=REGISTRY)
MET_ACTIVE_WORKERS = Gauge('visora_active_workers', 'Active background workers', registry=REGISTRY)

# initialize worker count (1 thread worker always)
MET_ACTIVE_WORKERS.set(1 if not USE_RQ else 0)

@app.route("/metrics")
def metrics():
    # expose prometheus metrics (standard)
    try:
        data = generate_latest(REGISTRY)
        return (data, 200, {'Content-Type': CONTENT_TYPE_LATEST})
    except Exception as e:
        log.exception("Failed to generate metrics")
        return jsonify({"error":"metrics failed","details":str(e)}), 500

# ------------------ Rate Limiting ------------------
# Default: 60 requests per minute per IP (tune if needed)
    app=app,
    key_func=get_remote_address,
    default_limits=["60 per minute"]
)

# Allow overriding via environment variable
custom_limit = os.getenv("API_RATE_LIMIT")
if custom_limit:
    limiter.limit(custom_limit)

# --------- Unified enqueue API (uses RQ if present, else thread fallback) ----------
def enqueue_render_job_internal(fn, *args, **kwargs):
    """
    Enqueue a render job (fn callable). Returns job_id string.
    """
    job_id = uuid.uuid4().hex
    render_jobs[job_id] = {"status":"queued","created_at": datetime.utcnow().isoformat()}
    MET_JOB_SUBMITTED.inc()

    if USE_RQ:
        try:
            # push to RQ
            rq_job = rq_queue.enqueue(fn, *args, **kwargs, job_id=job_id)
            render_jobs[job_id]["status"] = "queued_rq"
            render_jobs[job_id]["rq_id"] = rq_job.get_id()
            log.info("Enqueued job %s to RQ %s", job_id, rq_job.get_id())
        except Exception as e:
            log.exception("RQ enqueue failed, falling back to thread: %s", e)
            # fallback to thread
            enqueue_thread_job({"job_id":job_id,"fn":fn,"args":args,"kwargs":kwargs})
    else:
        enqueue_thread_job({"job_id":job_id,"fn":fn,"args":args,"kwargs":kwargs})

    return job_id

# Example wrapper for existing generate flow
@app.route("/enqueue_render", methods=["POST"])
@limiter.limit("10/minute")  # endpoint-specific stricter limit
def enqueue_render():
    """
    POST JSON:
      { "script":"text", "flow":"ucve" }
    Returns: {"job_id":"..."}
    """
    data = request.get_json() or {}
    script = data.get("script","")
    flow = data.get("flow","ucve")
    user_email = data.get("user_email","demo@visora.com")

    # create DB UserVideo record (simplified)
    video = UserVideo(user_email=user_email, title=f"Auto {datetime.utcnow().isoformat()}", script=script, template="UCVE", status="queued")
    db.session.add(video)
    db.session.commit()

    # Choose the actual function to run (safe wrappers)
    def job_wrapper(video_id, script_text, flow_name):
        try:
            # mark running in DB
            v = UserVideo.query.get(video_id)
            if v:
                v.status = "running"; db.session.commit()

            # Example: call existing UCVE flow
            if flow_name == "ucve" and "generate_cinematic_scene" in globals():
                out = generate_cinematic_scene(script_text)
            else:
                # fallback basic generator (should be replaced with real function)
                out = generate_cinematic_scene(script_text) if "generate_cinematic_scene" in globals() else None

            # update DB
            v = UserVideo.query.get(video_id)
            if v:
                v.status = "done"; v.file_path = out; db.session.commit()
            MET_JOB_DONE.inc()
            render_jobs[video_id.hex if hasattr(video_id,'hex') else video_id] = {"status":"done","output_file": out}
        except Exception as e:
            log.exception("Background job failed: %s", e)
            try:
                if v:
                    v.status = "failed"; db.session.commit()
            except:
                pass
            MET_JOB_FAILED.inc()

    # Enqueue (pass DB id)
    vid = video.id
    job_id = enqueue_render_job_internal(job_wrapper, vid, script, flow)
    return jsonify({"job_id": job_id, "video_db_id": vid})

# --------- Graceful shutdown handler ----------
shutdown_flag = False
def _graceful_shutdown(signum, frame):
    global shutdown_flag, thread_worker_running
    log.info("Graceful shutdown signal received: %s", signum)
    shutdown_flag = True
    # stop thread worker
    thread_worker_running = False
    # if using RQ, gracefully terminate is left to RQ worker process (not in this web process)
    # small wait for jobs cleanup
    time.sleep(1)
    log.info("Shutdown cleanup done.")

signal.signal(signal.SIGTERM, _graceful_shutdown)
signal.signal(signal.SIGINT, _graceful_shutdown)
atexit.register(lambda: log.info("Exiting visora backend."))

# --------- Health & readiness endpoints (expanded) ----------
@app.route("/readiness", methods=["GET"])
def readiness():
    # basic checks: DB OK, Redis optional, worker alive
    ok = True
    reasons = []
    try:
        # DB quick ping
        db.session.execute("SELECT 1")
    except Exception as e:
        ok = False
        reasons.append(f"db:{str(e)[:120]}")

    if USE_RQ:
        try:
            _ = redis_conn.ping()
        except Exception as e:
            ok = False
            reasons.append(f"redis:{str(e)[:120]}")

    # thread worker status
    if not thread_worker.is_alive() and not USE_RQ:
        ok = False
        reasons.append("thread_worker_down")

    return jsonify({"ready": ok, "issues": reasons})

@app.route("/liveness", methods=["GET"])
def liveness():
    return jsonify({"alive": True, "time": datetime.utcnow().isoformat()})

# --------- Docker & Compose hints (string for convenience) ----------
DOCKER_HELPER = """
# Dockerfile (suggested)
FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
# if you use redis + rq, ensure redis host available in REDIS_URL env
ENV FLASK_ENV=production
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "visora_backend:app", "--workers", "2", "--timeout", "120"]
"""

DOCKER_COMPOSE = """
# docker-compose.yml snippet
version: '3.7'
services:
  web:
    build: .
    ports: ['5000:8000']
    environment:
      - REDIS_URL=redis://redis:6379/0
      - DATABASE_URL=sqlite:///visora_data.db
  redis:
    image: redis:7-alpine
    restart: unless-stopped
"""

# End of UCVE v17 module

# ====================================================
# 🧩 UCVE v18: Final Production Layer (Profile + Gallery + Admin)
# ====================================================
from flask import send_file

# ---------- 1️⃣ Profile Photo Upload + Plan Update ----------
@app.route("/profile/photo", methods=["POST"])
def upload_profile_photo():
    user_email = request.form.get("email")
    if "photo" not in request.files or not user_email:
        return jsonify({"error": "email & photo required"}), 400
    f = request.files["photo"]
    if not allowed_file(f.filename, ALLOWED_IMAGE_EXT):
        return jsonify({"error": "invalid image"}), 400
    rel = save_upload(f, "profile_pics")
    u = UserProfile.query.filter_by(email=user_email).first()
    if not u:
        return jsonify({"error": "user not found"}), 404
    u.photo = rel
    db.session.commit()
    return jsonify({"status": "success", "photo": rel})

@app.route("/profile/update_plan", methods=["POST"])
def update_user_plan():
    data = request.get_json(force=True)
    email, plan, credits = data.get("email"), data.get("plan"), int(data.get("credits", 0))
    u = UserProfile.query.filter_by(email=email).first()
    if not u:
        return jsonify({"error": "user not found"}), 404
    u.plan = plan
    u.credits = u.credits + credits
    db.session.commit()
    return jsonify({"status": "updated", "plan": plan, "credits": u.credits})

# ---------- 2️⃣ Gallery Controls ----------
@app.route("/gallery/delete", methods=["POST"])
def gallery_delete():
    data = request.get_json(force=True)
    vid_id = data.get("video_id")
    v = UserVideo.query.get(vid_id)
    if not v:
        return jsonify({"error": "not found"}), 404
    try:
        if v.file_path:
            abs_path = _abs_path(v.file_path)
            if os.path.exists(abs_path):
                os.remove(abs_path)
        db.session.delete(v)
        db.session.commit()
        return jsonify({"status": "deleted", "video_id": vid_id})
    except Exception as e:
        log.exception("delete failed")
        return jsonify({"error": str(e)}), 500

@app.route("/gallery/download/<int:vid_id>")
def gallery_download(vid_id):
    v = UserVideo.query.get(vid_id)
    if not v or not v.file_path:
        return jsonify({"error": "not found"}), 404
    return send_file(_abs_path(v.file_path), as_attachment=True)

@app.route("/gallery/rename", methods=["POST"])
def gallery_rename():
    data = request.get_json(force=True)
    vid_id, new_title = data.get("video_id"), data.get("new_title")
    v = UserVideo.query.get(vid_id)
    if not v:
        return jsonify({"error": "not found"}), 404
    v.title = new_title
    db.session.commit()
    return jsonify({"status": "renamed", "title": new_title})

# ---------- 3️⃣ Admin APIs ----------
@app.route("/admin/users", methods=["GET"])
def admin_users():
    out = []
    for u in UserProfile.query.all():
        out.append({"email": u.email, "plan": u.plan, "credits": u.credits, "country": u.country})
    return jsonify(out)

@app.route("/admin/templates", methods=["GET", "POST", "DELETE"])
def admin_templates():
    if request.method == "GET":
        all_t = TemplateCatalog.query.all()
        return jsonify([{"id": t.id, "name": t.name, "category": t.category, "score": t.trending_score} for t in all_t])
    if request.method == "POST":
        d = request.get_json(force=True)
        t = TemplateCatalog(name=d.get("name"), category=d.get("category"), thumbnail=d.get("thumbnail"))
        db.session.add(t)
        db.session.commit()
        return jsonify({"status": "added", "id": t.id})
    if request.method == "DELETE":
        d = request.get_json(force=True)
        tid = d.get("id")
        t = TemplateCatalog.query.get(tid)
        if not t:
            return jsonify({"error": "not found"}), 404
        db.session.delete(t)
        db.session.commit()
        return jsonify({"status": "deleted"})

@app.route("/admin/credits", methods=["POST"])
def admin_add_credits():
    data = request.get_json(force=True)
    email, amount = data.get("email"), int(data.get("amount", 0))
    u = UserProfile.query.filter_by(email=email).first()
    if not u:
        return jsonify({"error": "user not found"}), 404
    u.credits += amount
    db.session.commit()
    return jsonify({"status": "credits_added", "credits": u.credits})

@app.route("/admin/jobs", methods=["GET"])
def admin_jobs():
    jobs = []
    for vid in UserVideo.query.order_by(UserVideo.created_at.desc()).limit(100).all():
        jobs.append({"id": vid.id, "user": vid.user_email, "title": vid.title, "status": vid.status, "created": vid.created_at.isoformat()})
    return jsonify(jobs)

# ---------- 4️⃣ Auto Cleaner ----------
@app.route("/admin/cleanup", methods=["POST"])
def admin_cleanup():
    now = time.time()
    removed = 0
    for folder in [TMP_FOLDER, OUTPUT_FOLDER]:
        for f in Path(folder).glob("*"):
            try:
                if f.is_file() and now - f.stat().st_mtime > 7 * 86400:
                    f.unlink()
                    removed += 1
            except Exception:
                pass
    return jsonify({"status": "cleaned", "files_removed": removed})

# ---------- 5️⃣ Thumbnail Generator ----------
def generate_video_thumbnail(video_path: str) -> str:
    """Generate first-frame thumbnail for gallery display."""
    thumb_path = str(OUTPUT_FOLDER / f"thumb_{uuid.uuid4().hex[:8]}.jpg")
    try:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(thumb_path, frame)
        cap.release()
    except Exception as e:
        log.warning("thumbnail failed: %s", e)
    return thumb_path
# ====================================================
# End of UCVE v18 module ✅

# ==========================================================
# UCVE v19.5 — Hybrid Character & Voice Sync Engine 🎙️
# Author: Aimantuvya & GPT-5
# Description:
# - Handles user voice uploads (single/multiple)
# - Auto-assigns voices to characters by gender/age/type
# - Fallbacks to AI-generated default voices if missing
# ==========================================================

import os
from pathlib import Path
from flask import request, jsonify

# Default AI fallback voices
AI_VOICES = {
    "male": "voices/ai_male.wav",
    "female": "voices/ai_female.wav",
    "child": "voices/ai_child.wav",
    "old": "voices/ai_old.wav",
    "general": "voices/ai_general.wav"
}


def match_character_voice(user_id, character_name):
    """Auto map user-provided or AI-generated voice based on keywords"""
    base_path = Path(f"voices/{user_id}")
    if not base_path.exists():
        base_path.mkdir(parents=True, exist_ok=True)

    voice_files = list(base_path.glob("*.wav"))
    name_lower = character_name.lower()

    # Helper to find a voice file matching a keyword
    def find_voice(keyword):
        for v in voice_files:
            if keyword in v.stem.lower():
                return str(v)
        return None

    # Match by type
    if "female" in name_lower or "queen" in name_lower or "girl" in name_lower:
        return find_voice("female") or AI_VOICES["female"]

    elif "male" in name_lower or "tiger" in name_lower or "king" in name_lower or "man" in name_lower:
        return find_voice("male") or AI_VOICES["male"]

    elif "child" in name_lower or "kid" in name_lower or "boy" in name_lower:
        return find_voice("child") or AI_VOICES["child"]

    elif "old" in name_lower or "grand" in name_lower or "elder" in name_lower:
        return find_voice("old") or AI_VOICES["old"]

    else:
        # fallback: use user’s first uploaded voice or AI general
        if voice_files:
            return str(voice_files[0])
        return AI_VOICES["general"]


@app.route("/voice/upload", methods=["POST"])
def upload_voice():
    """User uploads one or more voice samples"""
    email = request.form.get("email")
    if "voice" not in request.files:
        return jsonify({"error": "voice file missing"}), 400

    f = request.files["voice"]
    filename = f.filename.replace(" ", "_")
    user_folder = Path(f"voices/{email}")
    user_folder.mkdir(parents=True, exist_ok=True)

    file_path = user_folder / filename
    f.save(file_path)

    return jsonify({"status": "uploaded", "voice_path": str(file_path)})


@app.route("/character/upload", methods=["POST"])
def upload_character():
    """Upload character photo + auto assign matching voice"""
    email = request.form.get("email")
    char_name = request.form.get("character", "default")

    if "photo" not in request.files:
        return jsonify({"error": "photo required"}), 400

    f = request.files["photo"]
    char_folder = Path(f"characters/{email}")
    char_folder.mkdir(parents=True, exist_ok=True)

    rel_path = char_folder / f.filename.replace(" ", "_")
    f.save(rel_path)

    # Auto voice mapping
    voice_path = match_character_voice(email, char_name)

    # Save to DB (if enabled)
    try:
        if "CharacterProfile" in globals():
            c = CharacterProfile(email=email, name=char_name, photo=str(rel_path), voice_path=voice_path)
            db.session.add(c)
            db.session.commit()
    except Exception as e:
        log.warning(f"Character DB update failed: {e}")

    return jsonify({
        "status": "character_uploaded",
        "character": char_name,
        "photo": str(rel_path),
        "voice_assigned": voice_path
    })


@app.route("/ai/voices", methods=["GET"])
def list_ai_voices():
    """List all default Visora AI voices available"""
    return jsonify({"ai_voices": AI_VOICES})


@app.route("/voice/generate", methods=["POST"])
def generate_ai_voice():
    """Generate missing AI default voice placeholders"""
    for key, path in AI_VOICES.items():
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not out_path.exists():
            with open(out_path, "wb") as f:
                f.write(b"")  # placeholder - real TTS will fill later
    return jsonify({"status": "AI voice placeholders created"})

# ==========================================================
# UCVE v20 — Emotion Voice + Auto Lip Sync Engine 🎭
# Author: Aimantuvya & GPT-5
# Description:
# - Detects emotion from voice/script
# - Generates realistic lip-sync & expressions for characters
# - Integrates with UCVE cinematic video pipeline
# ==========================================================

import librosa
import numpy as np
import cv2
from moviepy.editor import ImageClip, AudioFileClip, CompositeAudioClip

# ---------- Emotion Detection ----------
def detect_emotion_from_audio(audio_path):
    """Analyze audio tone to detect emotion"""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        energy = np.mean(np.abs(y))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        if energy > 0.15 and tempo > 100:
            return "happy"
        elif energy < 0.05:
            return "sad"
        elif zcr > 0.12:
            return "angry"
        else:
            return "neutral"
    except Exception as e:
        log.warning(f"Emotion detect failed: {e}")
        return "neutral"


# ---------- Lip Sync + Expression ----------
def generate_lip_sync_video(photo_path, audio_path, output_path, emotion="neutral"):
    """Create talking photo with lip sync + emotion-driven motion"""
    img = cv2.imread(photo_path)
    if img is None:
        raise ValueError("Invalid image path")

    # Resize for consistent output
    img = cv2.resize(img, (1280, 720))
    base = ImageClip(photo_path).set_duration(AudioFileClip(audio_path).duration)

    # Add subtle motion based on emotion
    if emotion == "happy":
        motion_scale = 1.02
    elif emotion == "sad":
        motion_scale = 0.99
    elif emotion == "angry":
        motion_scale = 1.05
    else:
        motion_scale = 1.0

    # Fake mouth open-close animation using frame alternation
    clips = []
    dur = AudioFileClip(audio_path).duration
    step = 0.12
    t = 0.0
    toggle = False
    while t < dur:
        part = base.resize(motion_scale if toggle else 1.0)
        clips.append(part.set_duration(min(step, dur - t))
        t += step
        toggle = not toggle

    final = concatenate_videoclips(clips, method="compose")
    audio_clip = AudioFileClip(audio_path)
    final = final.set_audio(audio_clip)
    final.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac")

    return output_path


@app.route("/ai/lipsync", methods=["POST"])
def create_lipsync_scene():
    """
    POST form-data:
      - photo: character image
      - audio: character voice (wav/mp3)
      - emotion (optional)
    Output: Lip-synced video file path
    """
    if "photo" not in request.files or "audio" not in request.files:
        return jsonify({"error": "photo & audio required"}), 400

    photo = request.files["photo"]
    audio = request.files["audio"]
    emotion_input = request.form.get("emotion")

    photo_path = save_upload(photo, "lipsync_photos")
    audio_path = save_upload(audio, "lipsync_audio")

    # Auto-detect emotion if not provided
    emotion = emotion_input or detect_emotion_from_audio(_abs_path(audio_path))

    out_file = OUTPUT_FOLDER / f"lipsync_{uuid.uuid4().hex[:8]}.mp4"
    result_path = generate_lip_sync_video(_abs_path(photo_path), _abs_path(audio_path), str(out_file), emotion)

    return jsonify({
        "status": "lipsync_created",
        "emotion": emotion,
        "output": str(Path(result_path).relative_to(BASE_DIR))
    })


@app.route("/ai/emotion_detect", methods=["POST"])
def emotion_from_audio():
    """Standalone API for emotion analysis"""
    if "audio" not in request.files:
        return jsonify({"error": "audio file required"}), 400
    f = request.files["audio"]
    rel = save_upload(f, "emotion_audio")
    emotion = detect_emotion_from_audio(_abs_path(rel))
    return jsonify({"emotion": emotion, "audio": rel})

# ====================================================
# UCVE v21 — Real-Time Emotion Morphing & Scene Fusion
# ====================================================
import math
import numpy as np
import soundfile as sf
import scipy.signal as sps
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip, vfx

# --- Simple voice morph by pitch-shifting + amplitude envelope shaping ---
def pitch_shift_np(y: np.ndarray, sr: int, n_steps: float):
    """Simple pitch shift via resampling (not high-quality but fast). n_steps semitones."""
    # factor = 2^(n_steps/12)
    factor = 2.0 ** (n_steps / 12.0)
    # resample to new length then resample back to original sr
    new_len = int(len(y) / factor)
    y2 = sps.resample(y, new_len)
    y_back = sps.resample(y2, len(y))
    return y_back

def envelope_shape(y: np.ndarray, strength: float = 1.0):
    """Apply mild dynamic shaping — compress/expand via simple tanh curve."""
    # normalize
    maxv = max(1e-9, np.max(np.abs(y))
    norm = y / maxv
    shaped = np.tanh(norm * (1.0 + 0.8 * strength))
    return (shaped * maxv).astype(y.dtype)

def morph_voice_emotion(audio_path: str, out_path: str, target_emotion: str = "neutral"):
    """
    Read audio file, morph pitch/amplitude according to target_emotion,
    and write out a new audio file (wav).
    target_emotion ∈ {"happy","sad","angry","motivational","neutral"}
    """
    data, sr = sf.read(audio_path)
    # if stereo, average to mono for processing
    if data.ndim == 2:
        mono = data.mean(axis=1)
    else:
        mono = data

    # default params
    semitone_shift = 0.0
    amp_strength = 0.0

    if target_emotion == "happy":
        semitone_shift = +2.5   # slight pitch up
        amp_strength = +0.2     # slight expansion
    elif target_emotion == "sad":
        semitone_shift = -2.5
        amp_strength = -0.25
    elif target_emotion == "angry":
        semitone_shift = +1.2
        amp_strength = +0.6
    elif target_emotion == "motivational":
        semitone_shift = +1.8
        amp_strength = +0.35
    else:
        semitone_shift = 0.0
        amp_strength = 0.0

    # pitch shift
    try:
        shifted = pitch_shift_np(mono, sr, semitone_shift)
    except Exception:
        shifted = mono

    # apply envelope shaping / dynamic
    shaped = envelope_shape(shifted, amp_strength)

    # simple limiter for angry (harder)
    if target_emotion == "angry":
        shaped = np.clip(shaped, -0.9 * np.max(np.abs(shaped)), 0.9 * np.max(np.abs(shaped))

    # convert back to stereo if original was stereo
    if data.ndim == 2:
        shaped = np.stack([shaped, shaped], axis=1)

    sf.write(out_path, shaped, sr)
    return out_path

# --- Simple scene fusion / transitions ---
def fuse_scenes_timeline(scene_paths: list, transitions: list = None, final_out: str = None):
    """
    scene_paths: list of local video file paths in order
    transitions: list of transition types between scenes (len = len(scene_paths)-1), options: 'crossfade','cut','speed_ramp'
    Returns: path to fused video
    """
    if not transitions:
        transitions = ["crossfade"] * max(0, len(scene_paths)-1)

    clips = []
    for p in scene_paths:
        c = VideoFileClip(str(p))
        clips.append(c)

    # apply transitions
    fused = None
    try:
        if not clips:
            raise ValueError("No clips provided")
        # build progressive fusion
        out_clips = []
        for i, clip in enumerate(clips):
            if i == 0:
                out_clips.append(clip)
                continue
            ttype = transitions[i-1] if i-1 < len(transitions) else "crossfade"
            prev = out_clips[-1]
            if ttype == "cut":
                out_clips.append(clip)
            elif ttype == "speed_ramp":
                # speed ramp: slow last 0.6s of prev and speed up first 0.6s of next
                ramp = 0.6
                # clamp ramp to clip durations
                ramp = min(ramp, prev.duration/2, clip.duration/2)
                prev_slow = prev.fx(vfx.speedx, 0.9).subclip(max(0, prev.duration - ramp), prev.duration)
                next_fast = clip.fx(vfx.speedx, 1.08).subclip(0, min(ramp, clip.duration))
                middle = concatenate_videoclips([prev.subclip(0, prev.duration - ramp), prev_slow, next_fast, clip.subclip(min(ramp, clip.duration), clip.duration)], method="compose")
                # replace last with middle
                out_clips[-1] = middle
            else:  # crossfade default
                # crossfade of 0.6s (safe)
                fused_pair = concatenate_videoclips([prev, clip.crossfadein(0.6)], method="compose")
                out_clips[-1] = fused_pair
        # if out_clips has one big clip already
        if len(out_clips) == 1:
            fused = out_clips[0]
        else:
            fused = concatenate_videoclips(out_clips, method="compose")
    except Exception as e:
        log.exception("Scene fusion failed: %s", e)
        # fallback: try naive concat
        fused = concatenate_videoclips(clips, method="compose")

    out_name = final_out or (OUTPUT_FOLDER / f"fused_{uuid.uuid4().hex[:8]}.mp4")
    fused.write_videofile(str(out_name), fps=24, codec="libx264", audio_codec="aac", threads=2, logger=None)
    # close clips
    try:
        fused.close()
        for c in clips:
            c.close()
    except Exception:
        pass
    return str(out_name)

# ---------------- Flask endpoints ----------------
@app.route("/ai/morph_emotion", methods=["POST"])
def api_morph_emotion():
    """
    Form-data:
      - audio (file) OR audio_path (string)
      - target_emotion: happy/sad/angry/motivational/neutral
    Returns:
      { "status":"ok","morphed_audio": "<path>" }
    """
    target = (request.form.get("target_emotion") or "neutral").lower()
    if "audio" in request.files:
        a = request.files["audio"]
        tmp = TMP_FOLDER / f"morph_src_{uuid.uuid4().hex[:8]}.wav"
        a.save(tmp)
        src = str(tmp)
    else:
        src = request.form.get("audio_path")
        if not src:
            return jsonify({"error":"audio required"}), 400
        src = _abs_path(src)

    outp = OUTPUT_FOLDER / f"morphed_{target}_{uuid.uuid4().hex[:8]}.wav"
    try:
        morphed = morph_voice_emotion(src, str(outp), target_emotion=target)
        return jsonify({"status":"ok","morphed_audio": str(Path(morphed).relative_to(BASE_DIR))})
    except Exception as e:
        log.exception("morph failed: %s", e)
        return jsonify({"error":"morph failed","details":str(e)}), 500

@app.route("/scene/fuse", methods=["POST"])
def api_scene_fuse():
    """
    POST JSON:
      { "scenes": ["outputs/scene1.mp4","outputs/scene2.mp4"], "transitions": ["crossfade"] }
    Returns:
      { "status":"ok","file":"outputs/fused_xxx.mp4" }
    """
    data = request.get_json() or {}
    scenes = data.get("scenes", [])
    transitions = data.get("transitions", [])
    if not scenes:
        return jsonify({"error":"scenes required"}), 400
    abs_scenes = [ _abs_path(s) for s in scenes ]
    try:
        out = fuse_scenes_timeline(abs_scenes, transitions)
        # optionally upload to firebase
        cloud = None
        if FIREBASE_BUCKET:
            try:
                cloud = upload_to_firebase(out)
            except Exception:
                cloud = None
        return jsonify({"status":"ok","file": str(Path(out).relative_to(BASE_DIR)), "cloud_url": cloud})
    except Exception as e:
        log.exception("scene fuse failed: %s", e)
        return jsonify({"error":"fuse failed","details":str(e)}), 500

# ====================================================
# UCVE v22 — Character–Voice Preset Management System
# ====================================================

@app.route("/preset/save", methods=["POST"])
def save_user_preset():
    """
    POST JSON:
    {
      "user_email": "demo@visora.com",
      "preset_name": "My Jungle Style",
      "characters": [
        {"name": "Tiger", "photo": "characters/tiger.jpg", "voice": "voices/male1.wav", "emotion": "angry"},
        {"name": "Monkey", "photo": "characters/monkey.jpg", "voice": "voices/funny.wav", "emotion": "happy"}
      ],
      "voice_style": "cinematic",
      "quality": "4K"
    }
    """
    data = request.get_json(force=True)
    email = data.get("user_email")
    if not email:
        return jsonify({"error": "user_email required"}), 400

    preset = UserPreset.query.filter_by(user_email=email).first()
    if not preset:
        preset = UserPreset(user_email=email)

    preset.preset_name = data.get("preset_name", "default")
    preset.characters_json = json.dumps(data.get("characters", []))
    preset.voice_style = data.get("voice_style", "default")
    preset.quality = data.get("quality", "HD")
    preset.last_used = datetime.utcnow()

    db.session.add(preset)
    db.session.commit()

    return jsonify({"status": "preset_saved", "preset_name": preset.preset_name})


@app.route("/preset/load", methods=["GET"])
def load_user_preset():
    """Load saved preset for user"""
    email = request.args.get("user_email")
    if not email:
        return jsonify({"error": "user_email required"}), 400

    preset = UserPreset.query.filter_by(user_email=email).first()
    if not preset:
        return jsonify({"message": "no preset found"}), 404

    return jsonify({
        "preset_name": preset.preset_name,
        "characters": json.loads(preset.characters_json or "[]"),
        "voice_style": preset.voice_style,
        "quality": preset.quality,
        "last_used": preset.last_used.isoformat()
    })


@app.route("/preset/delete", methods=["POST"])
def delete_user_preset():
    """Delete a user's preset"""
    data = request.get_json(force=True)
    email = data.get("user_email")
    if not email:
        return jsonify({"error": "user_email required"}), 400
    preset = UserPreset.query.filter_by(user_email=email).first()
    if not preset:
        return jsonify({"error": "not found"}), 404
    db.session.delete(preset)
    db.session.commit()
    return jsonify({"status": "preset_deleted"})

# ===============================================================
# 🔍 SYSTEM HEALTH CHECK ROUTE
# ===============================================================

@app.route("/selfcheck", methods=["GET"])
def selfcheck():
    import importlib
    modules = [
        "flask", "torch", "open3d", "stripe",
        "moviepy", "firebase_admin", "prometheus_client"
    ]
    status = {}

    for m in modules:
        try:
            importlib.import_module(m)
            status[m] = "✅ Loaded"
        except Exception as e:
            status[m] = f"❌ Missing ({str(e)})"

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        status["torch_device"] = f"🧠 Torch running on {device}"
    except:
        status["torch_device"] = "⚠️ Torch not loaded"

    return jsonify({
        "app": "Visora AI Backend UCVE v22",
        "status": "✅ OK",
        "modules": status,
        "uptime": str(datetime.datetime.now())
    })

# ===============================================================
# 🎙️ VFE UCVE v23 - Voice Fusion Engine (AI Narration + Sync)
# ===============================================================
import io
import base64
from gtts import gTTS
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip

@app.route("/generate_voice_video", methods=["POST"])
def generate_voice_video():
    """
    Generate video with AI narration (text-to-speech + auto sync)
    """
    try:
        data = request.get_json(force=True)
        script_text = data.get("script", "")
        video_path = data.get("video_path", "")

        if not script_text:
            return jsonify({"status": "error", "message": "No script text provided."}), 400
        if not video_path or not os.path.exists(video_path):
            return jsonify({"status": "error", "message": "Video file not found."}), 400

        # 🎤 Generate AI voice
        tts = gTTS(text=script_text, lang="en", slow=False)
        audio_file = os.path.join(RENDER_PATH, f"{uuid.uuid4()}.mp3")
        tts.save(audio_file)

        # 🎬 Merge voice + video
        video_clip = VideoFileClip(video_path)
        audio_clip = AudioFileClip(audio_file)
        final_clip = video_clip.set_audio(audio_clip)

        output_path = os.path.join(RENDER_PATH, f"vfe_{uuid.uuid4()}.mp4")
        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

        return jsonify({
            "status": "success",
            "message": "AI Voice Fusion video generated successfully.",
            "output": output_path
        })

    except Exception as e:
        log.exception("VFE UCVE v23 error")
        return jsonify({"status": "error", "message": str(e)}), 500

# ===============================================================
# 🖼️ SceneGen UCVE v24 - Scene & Background Generator (short/long)
# ===============================================================
import math
import random
import requests
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageSequenceClip, AudioFileClip, VideoFileClip
from typing import List

IMAGE_API_KEY = os.getenv("IMAGE_API_KEY", None)  # optional (Replicate / SD)
IMAGE_API_URL = os.getenv("IMAGE_API_URL", "")    # optional endpoint if using remote API
# default frame rate and durations
SHORT_DURATION = 8    # seconds for short video
LONG_DURATION = 45    # seconds for long video
FPS = 15

def _prompt_from_script(script: str, n_scenes: int) -> List[str]:
    """Simple scene prompt extractor — splits script into n_scenes chunks and returns short prompts."""
    words = script.split()
    if len(words) < 5:
        return [script] * n_scenes
    per = max(1, len(words)//n_scenes)
    prompts = []
    for i in range(n_scenes):
        chunk = " ".join(words[i*per:(i+1)*per])
        prompts.append(chunk.strip() or script[:50])
    return prompts

def _generate_image_via_api(prompt: str, out_path: str) -> bool:
    """Optional: call external image generation API (Replicate / Stable Diffusion)."""
    if not IMAGE_API_KEY or not IMAGE_API_URL:
        return False
    try:
        payload = {"prompt": prompt, "width": 1024, "height": 576}
        headers = {"Authorization": f"Bearer {IMAGE_API_KEY}"}
        r = requests.post(IMAGE_API_URL, json=payload, headers=headers, timeout=120)
        r.raise_for_status()
        data = r.json()
        # Expect base64 or url in data; adapt as your API returns
        if "image_base64" in data:
            b = base64.b64decode(data["image_base64"])
            with open(out_path, "wb") as f:
                f.write(b)
            return True
        elif "image_url" in data:
            rr = requests.get(data["image_url"], timeout=60)
            rr.raise_for_status()
            with open(out_path, "wb") as f:
                f.write(rr.content)
            return True
    except Exception as e:
        log.warning("External image API failed: %s", e)
    return False

def _generate_fallback_image(prompt: str, out_path: str, size=(1280,720)):
    """Fallback: make a simple stylized background with the prompt text."""
    w,h = size
    img = Image.new("RGB", size, (20+random.randint(0,60), 20+random.randint(0,60), 40+random.randint(0,60))
    draw = ImageDraw.Draw(img)
    # basic gradient
    for i in range(h):
        r = int((i/h)*40) + 20
        g = int((1 - i/h)*40) + 20
        b = int(((i/h)*i/h)*60) + 20
        draw.line([(0,i),(w,i)], fill=(r,g,b))
    # draw prompt text in center (small)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 26)
    except:
        font = ImageFont.load_default()
    text = (prompt[:200] + "...") if len(prompt) > 200 else prompt
    tw, th = draw.textsize(text, font=font)
    draw.text(((w-tw)/2, (h-th)/2), text, fill=(255,255,255), font=font)
    img.save(out_path)

def _make_scene_images(prompts: List[str], tmpdir: str) -> List[str]:
    """For each prompt, try API then fallback; return list of image paths."""
    image_paths = []
    for i,p in enumerate(prompts):
        fname = os.path.join(tmpdir, f"scene_{i:03d}.png")
        ok = _generate_image_via_api(p, fname)
        if not ok:
            _generate_fallback_image(p, fname, size=(1280,720))
        image_paths.append(fname)
    return image_paths

def _frames_from_images(images: List[str], fps: int, duration: int) -> List[str]:
    """Convert scene images into a sequence of frames (static holds with simple crossfade)"""
    frames = []
    per_scene = max(1, int(math.ceil((duration / max(1, len(images)) * fps))
    for idx,img in enumerate(images):
        # simple duplicate frames for static hold (could implement gradual transition)
        for k in range(per_scene):
            frames.append(img)
    return frames

@app.route("/scenegen", methods=["POST"])
def scenegen_endpoint():
    """
    POST JSON:
    {
      "script": "...",
      "type": "short" | "long",   # optional; default short
      "n_scenes": 4              # optional
    }
    """
    try:
        data = request.get_json(force=True) or {}
        script = data.get("script", "").strip()
        vtype = data.get("type", "short")
        n_scenes = int(data.get("n_scenes", 4))

        if not script:
            return jsonify({"status": "error", "message": "No script provided."}), 400
        # decide duration
        duration = SHORT_DURATION if vtype == "short" else LONG_DURATION
        # prepare prompts
        prompts = _prompt_from_script(script, n_scenes)

        # tmpdir
        tmpdir = os.path.join(RENDER_PATH, f"scenegen_{uuid.uuid4()}")
        os.makedirs(tmpdir, exist_ok=True)

        # generate scene images
        images = _make_scene_images(prompts, tmpdir)

        # convert to frames (simple hold)
        frames = _frames_from_images(images, FPS, duration)

        # build video clip from frames
        clip = ImageSequenceClip(frames, fps=FPS)
        out_path = os.path.join(RENDER_PATH, f"scenegen_{uuid.uuid4()}.mp4")
        clip.write_videofile(out_path, codec="libx264", audio=False)

        return jsonify({"status": "success", "output": out_path, "tmpdir": tmpdir})

    except Exception as e:
        log.exception("SceneGen UCVE v24 failed")
        return jsonify({"status": "error", "message": str(e)}), 500

# ===============================================================
# 🚀 UCVE v25 - Auto Publisher (YouTube + Instagram + Facebook)
# ===============================================================
# Description:
# - Generate AI metadata (title, desc, thumbnail)
# - Allow upload to selected platforms
# - API token-based authentication (user-supplied)

from flask import Flask, request, jsonify
import os, json, uuid, datetime, logging as log
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
from typing import Optional

# 👇 ye line yahaan add karni hai
from visora_auto_publisher import generate_auto_meta, upload_to_platforms

def generate_auto_meta(video_path: str):
    """Generate auto title, description, and thumbnail."""
    title = f"Cinematic AI Creation {''.join(random.choices(string.ascii_uppercase, k=4))}"
    description = (
        "🎬 Created using Visora AI\n"
        "#VisoraAI #Cinematic #AIArt #Innovation"
    )

    # generate thumbnail (1st frame)
    try:
        clip = VideoFileClip(video_path)
        thumb_path = video_path.replace(".mp4", "_thumb.jpg")
        clip.save_frame(thumb_path, t=0.5)
        clip.close()
    except Exception as e:
        thumb_path = None
        print(f"⚠️ Thumbnail generation failed: {e}")

    return title, description, thumb_path


def upload_to_platforms(video_path, title, description, thumb, upload_options, user_tokens):
    """Upload to YouTube, Instagram, Facebook (selected by user)."""
    try:
        print("⏳ Starting multi-platform upload...")
        result = {}

        if upload_options.get("youtube"):
            print("🎥 Uploading to YouTube...")
            result["youtube"] = upload_youtube(video_path, title, description, thumb, user_tokens["youtube"])

        if upload_options.get("instagram"):
            print("📱 Uploading to Instagram...")
            result["instagram"] = upload_instagram(video_path, description, user_tokens["instagram"])

        if upload_options.get("facebook"):
            print("🌐 Uploading to Facebook...")
            result["facebook"] = upload_facebook(video_path, title, description, user_tokens["facebook"])

        print("✅ Upload complete:", result)
        return {"status": "success", "details": result}

    except Exception as e:
        print(f"⚠️ Upload failed: {e}")
        return {"status": "error", "message": str(e)}

# ===========================
# UCVE REALMODE ENGINE v3
# Lip-sync + Expression Motion (add at end of visora_backend.py)
# ===========================

import librosa
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, TextClip, vfx
from typing import Optional

def analyze_voice_for_lipsync(voice_path: str):
    """
    Return timestamps (seconds) where voice RMS energy peaks occur.
    Used to estimate mouth movement frames.
    """
    try:
        y, sr = librosa.load(voice_path, sr=None)
        # RMS (root-mean-square) energy
        rms = librosa.feature.rms(y=y)[0]
        # threshold = 75th percentile energy
        thresh = np.percentile(rms, 75)
        peaks = np.where(rms > thresh)[0]
        timestamps = librosa.frames_to_time(peaks, sr=sr)
        return timestamps.tolist()
    except Exception as e:
        # Return empty list on error
        return []

def generate_emotion_tone(voice_path: Optional[str]) -> str:
    """
    Lightweight placeholder: determine simple 'emotion' based on average volume.
    (This is a simple heuristic, not a neural emotion model.)
    """
    try:
        if not voice_path or not os.path.exists(voice_path):
            return "neutral"
        y, sr = librosa.load(voice_path, sr=None)
        avg_energy = float(np.mean(librosa.feature.rms(y=y)[0]))
        if avg_energy > 0.06:
            return "angry"
        if avg_energy > 0.03:
            return "happy"
        return "calm"
    except Exception:
        return "neutral"

def apply_realmode_v3(video_path: str, voice_path: Optional[str] = None):
    """
    Apply basic color enhancement + lip-sync overlay + emotion overlay.
    Returns dict with status & output path or error.
    """
    try:
        emotion = generate_emotion_tone(voice_path) if voice_path else "neutral"
        clip = VideoFileClip(video_path)

        # simple color/contrast enhancement
        try:
            clip = clip.fx(vfx.colorx, 1.12).fx(vfx.lum_contrast, lum=6, contrast=12)
        except Exception:
            pass

        # lip-sync timestamps (simple)
        mouth_motion = analyze_voice_for_lipsync(voice_path) if voice_path else []

        # small overlay labels (for debugging / visualization)
        mouth_label = TextClip(
            f"LipFrames: {len(mouth_motion)}", fontsize=22, color='yellow', bg_color='black'
        ).set_duration(clip.duration).set_position(("center", "bottom"))

        emotion_label = TextClip(
            f"Emotion: {emotion}", fontsize=22, color='white', bg_color='black'
        ).set_duration(clip.duration).set_position(("center", "top"))

        final = CompositeVideoClip([clip, mouth_label, emotion_label])

        # attach voice track if provided
        if voice_path and os.path.exists(voice_path):
            audio = AudioFileClip(voice_path)
            # if audio longer than clip, cut; else set_audio directly
            if audio.duration > final.duration:
                audio = audio.subclip(0, final.duration)
            final = final.set_audio(audio)

        # output file
        out_name = f"realmode_v3_{uuid.uuid4().hex[:8]}.mp4"
        output_path = os.path.join(RENDER_PATH, out_name)
        # write (use small preset for speed)
        final.write_videofile(output_path, codec="libx264", audio_codec="aac", threads=0, preset="fast")

        # close resources
        final.close()
        clip.close()
        if 'audio' in locals():
            audio.close()

        return {"status": "success", "emotion": emotion, "mouth_frames": len(mouth_motion), "output": output_path}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Route to use the RealMode v3 engine
@app.route("/realmode_v3", methods=["POST"])
def realmode_v3():
    try:
        if 'video' not in request.files:
            return jsonify({"error": "Video file missing"}), 400
        video = request.files['video']
        video_path = os.path.join(RENDER_PATH, f"input_v3_{uuid.uuid4().hex[:6]}.mp4")
        video.save(video_path)

        voice_path = None
        if 'voice' in request.files:
            voice = request.files['voice']
            voice_path = os.path.join(RENDER_PATH, f"voice_v3_{uuid.uuid4().hex[:6]}.mp3")
            voice.save(voice_path)

        result = apply_realmode_v3(video_path, voice_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# ===========================
# 📡 Individual Upload APIs
# ===========================

def upload_youtube(video_path, title, description, thumb, token):
    """YouTube upload (through user token)."""
    # placeholder pseudo logic (YouTube Data API)
    return f"Uploaded to YouTube ({os.path.basename(video_path)})"


def upload_instagram(video_path, caption, token):
    """Instagram upload (Reel/Feed)."""
    # placeholder pseudo logic (Instagram Graph API)
    return f"Uploaded to Instagram ({os.path.basename(video_path)})"


def upload_facebook(video_path, title, description, token):
    """Facebook video upload."""
    # placeholder pseudo logic (Facebook Graph API)
    return f"Uploaded to Facebook ({os.path.basename(video_path)})"

# =====================================================
# ☁️ UCVE v30 - CloudSafe Mode (100% Render Compatible)
# =====================================================
from flask import Flask, request, jsonify
import os, datetime, uuid

app = Flask(__name__)

# -----------------------------------------------------
# 🧠 Health Route
# -----------------------------------------------------
@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "✅ Visora-AI CloudSafe Backend Running",
        "version": "UCVE v30",
        "timestamp": str(datetime.datetime.now())
    }), 200


# -----------------------------------------------------
# 🎥 Simulated Video Render Endpoint
# -----------------------------------------------------
@app.route("/render", methods=["POST"])
def render_job():
    try:
        data = request.get_json(force=True)
        job_id = str(uuid.uuid4())
        print(f"🎞️ Render started for job {job_id}")

        return jsonify({
            "job_id": job_id,
            "status": "completed",
            "message": "Video render simulation success"
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# -----------------------------------------------------
# 🚀 Main (Render Safe Entry Point)
# -----------------------------------------------------
if __name__ == "__main__":
    # Use waitress (safer than gunicorn for free cloud)
    from waitress import serve
    port = int(os.environ.get("PORT", 5000))
    print(f"✅ Running UCVE v30 CloudSafe Mode on port {port}")
    serve(app, host="0.0.0.0", port=port)
