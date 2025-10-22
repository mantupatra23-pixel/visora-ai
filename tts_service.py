#!/usr/bin/env python3
"""
tts_service.py — AI Voice Generator (Stage-3)
Author: Aimantuvya & GPT-5
Description:
    Converts generated story/script text into realistic speech.
    Supports both ElevenLabs (if key available) and fallback Google TTS (gTTS).
"""

import os
import uuid
from gtts import gTTS

# Try ElevenLabs import
try:
    pass
    import requests
except ImportError:
    requests = None


# 🎙️ Fallback using gTTS
def _fallback_gtts(text, lang="en"):
    """
    Fallback Google TTS (works offline with internet)
    """
    if not text.strip():
        raise ValueError("No text provided for TTS")

    output_dir = os.getenv("UPLOAD_FOLDER", "uploads")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"tts_{uuid.uuid4().hex[:8]}.mp3")

    tts = gTTS(text=text, lang=lang)
    tts.save(out_path)
    print(f"✅ gTTS audio saved: {out_path}")
    return out_path


# ⚙️ ElevenLabs TTS (if available)
def _elevenlabs_tts(text, voice_id=None):
    """
    Generate voice using ElevenLabs API if keys available.
    """
    ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
    if not ELEVEN_API_KEY or not requests:
        print("⚠️ ElevenLabs key not found or requests module missing, using fallback.")
        return None

    voice_id = voice_id or os.getenv("ELEVEN_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
    api_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVEN_API_KEY
    }
    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.4,
            "similarity_boost": 0.9}}

    output_dir = os.getenv("UPLOAD_FOLDER", "uploads")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"tts_{uuid.uuid4().hex[:8]}.mp3")

    try:
        response = requests.post(api_url, headers=headers, json=payload)
        if response.status_code == 200:
            with open(out_path, "wb") as f:
                f.write(response.content)
            print(f"✅ ElevenLabs voice generated: {out_path}")
            return out_path
        else:
            print(
                f"❌ ElevenLabs failed: {
                    response.status_code}, using fallback.")
            return None
    except Exception as e:
        print("⚠️ ElevenLabs error:", e)
        return None


# 🚀 Public main function
def generate_and_save_tts(text: str, lang="en"):
    """
    Main function: text -> TTS file path
    """
    text = text.strip()
    if not text:
        raise ValueError("Text is empty for TTS")

    # 1. Try ElevenLabs if key available
    out_path = _elevenlabs_tts(text)
    if out_path:
        return out_path

    # 2. Fallback to gTTS
    return _fallback_gtts(text, lang)


# 🧪 Local test
if __name__ == "__main__":
    text = input("🎙️ Enter a short text for TTS: ")
    path = generate_and_save_tts(text)
    print(f"✅ Generated file path: {path}")
