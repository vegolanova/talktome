from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
import os
import sys


def main():
    # Load environment variables
    load_dotenv()

    # Get API credentials
    api_key = os.getenv("ELEVENLABS_API_KEY")
    voice_id = os.getenv("VOICE_API_KEY")  # Fixed variable name

    # Validate environment variables
    if not api_key:
        print("Error: ELEVENLABS_API_KEY not found in environment variables")
        sys.exit(1)

    if not voice_id:
        print("Error: VOICE_ID not found in environment variables")
        sys.exit(1)

    # Initialize ElevenLabs client
    try:
        elevenlabs = ElevenLabs(api_key=api_key)
    except Exception as e:
        print(f"Error initializing ElevenLabs client: {e}")
        sys.exit(1)

    # Speech-to-Text
    audio_file_path = "recorded.wav"
    transcribed_text = ""  # Initialize variable outside try block

    try:
        # Check if audio file exists
        if not os.path.exists(audio_file_path):
            print(f"Warning: Audio file '{audio_file_path}' not found, skipping transcription")
        else:
            print("Transcribing audio...")
            with open(audio_file_path, "rb") as audio_file:
                transcript = elevenlabs.speech_to_text.from_file(
                    file=audio_file,
                    model_id="scribe_v1"
                )

            transcribed_text = transcript.get("text", "")
            print(f"Transcription: {transcribed_text}")

    except Exception as e:
        print(f"Error during transcription: {e}")
        print("Continuing with default text for TTS...")
        # transcribed_text remains empty string

    # Text-to-Speech
    # You can use the transcribed text or provide your own
    text_to_speak = transcribed_text if transcribed_text else "Hello my name is Shrek. V is a monster hahaha and Emmanuel is super cool"

    try:
        print("Generating speech...")
        audio_stream = elevenlabs.text_to_speech.stream(
            text=text_to_speak,
            voice_id=voice_id,
            model_id="eleven_multilingual_v2"
        )

        # Save to a file
        output_file = "output.mp3"
        with open(output_file, "wb") as f:
            for chunk in audio_stream:
                if isinstance(chunk, bytes):
                    f.write(chunk)

        print(f"Audio saved to {output_file}")

    except Exception as e:
        print(f"Error during text-to-speech: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()