#!/usr/bin/env python3
"""
story_writer.py — AI Story + Dialogue Generator Module (Stage-3 Base)
Author: Aimantuvya & GPT-5
Description:
    Lightweight cinematic story generator that auto creates storylines 
    and basic dialogues using GPT-like pattern fallback.
"""

import os
import random
import textwrap

# Optional: If OPENAI_API_KEY exists, use OpenAI API
try:
    import openai
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if OPENAI_API_KEY:
        openai.api_key = OPENAI_API_KEY
except Exception:
    openai = None

# 🔮 Basic fallback story generator (offline mode)
def _local_story(topic: str, duration="short", style="cinematic"):
    """
    Fallback story generator without external API.
    Returns a cinematic-style short story with dialogues.
    """
    moods = ["dramatic", "motivational", "emotional", "thrilling", "peaceful"]
    mood = random.choice(moods)
    characters = ["Ayaan", "Meera", "Kabir", "Riya", "Aarav", "Tara"]
    char1, char2 = random.sample(characters, 2)

    story = f"""
🎬 *Title:* {topic.title()} — A {mood} {style} Story

Scene 1:
{char1}: "Have you ever wondered what {topic.lower()} really means?"
{char2}: "Every day I ask myself that question, but never find the answer."

Scene 2:
The camera pans slowly... wind rustles... a beam of light falls on {char1}'s face.

{char1}: "Maybe {topic.lower()} isn't something we find. Maybe it's something we become."
{char2}: "Then let's become it... together."

🎞️ End Scene.
"""
    return textwrap.dedent(story)

# 🚀 Primary function
def generate_story(topic: str, duration="short", style="cinematic"):
    """
    Generates a cinematic short story for a given topic.
    If OpenAI API key is present, uses GPT; otherwise fallback to local.
    """
    topic = topic.strip().capitalize()

    # Try API if available
    if openai and os.getenv("OPENAI_API_KEY"):
        try:
            print(f"🎯 Using OpenAI API for story generation: topic={topic}")
            prompt = f"Write a {style} short film script about '{topic}' with dialogues."
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=600
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            print("⚠️ OpenAI API failed, fallback mode:", e)

    # Fallback offline generator
    print(f"🌀 Using local fallback generator for topic: {topic}")
    return _local_story(topic, duration, style)


# 🧪 Local test
if __name__ == "__main__":
    t = input("🎬 Enter topic: ")
    print(generate_story(t))
