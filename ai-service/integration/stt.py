# example.py
import os
from dotenv import load_dotenv
from io import BytesIO
from elevenlabs.client import ElevenLabs

load_dotenv()

elevenlabs = ElevenLabs(
    api_key=os.getenv("ELEVENLABS_API_KEY"),
)

# Read local audio file
with open("recorded.wav", "rb") as f:
    audio_data = BytesIO(f.read())

transcription = elevenlabs.speech_to_text.convert(
    file=audio_data,
    model_id="scribe_v1",
    language_code="eng",
)

print(transcription.text)
