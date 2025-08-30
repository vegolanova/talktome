# stt.py
import os
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from io import BytesIO
import sounddevice as sd
import wavio

load_dotenv()

elevenlabs = ElevenLabs(
    api_key=os.getenv("ELEVENLABS_API_KEY"),
)

def get_transcript():
    # Record audio
    duration = 5  # seconds
    sample_rate = 44100
    print("Recording...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    print("Recording finished.")

    # Save to file (optional)
    wavio.write("recorded.wav", audio_data, sample_rate, sampwidth=2)

    # Read back for ElevenLabs
    with open("recorded.wav", "rb") as f:
        audio_bytes = BytesIO(f.read())

    transcription = elevenlabs.speech_to_text.convert(
        file=audio_bytes,
        model_id="scribe_v1",
        language_code="eng",
    )

    return transcription.text

print(get_transcript())