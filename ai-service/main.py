import os
import sys
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
import pygame

# --- CORRECTED IMPORTS FOR YOUR FOLDER STRUCTURE ---
from rag.document_loader import ScriptLoader
from rag.rag_pipeline import create_rag_chain
from integration.input import get_transcript
# ---------------------------------------------------

def generate_and_play_audio(client: ElevenLabs, text: str, voice_id: str, output_filename="output.mp3"):
    """
    Converts text to speech using ElevenLabs and plays the resulting audio file.
    """
    try:
        print("Shrek is thinking...")
        # Generate audio stream from the text
        audio_stream = client.text_to_speech.stream(
            text=text,
            voice_id=voice_id,
            model_id="eleven_multilingual_v2"
        )

        # Write the audio stream to a file
        with open(output_filename, "wb") as f:
            for chunk in audio_stream:
                if isinstance(chunk, bytes):
                    f.write(chunk)

        print(f"Playing Shrek's response...")
        # Initialize pygame mixer
        pygame.mixer.init()
        # Load the audio file
        pygame.mixer.music.load(output_filename)
        # Play the audio
        pygame.mixer.music.play()
        # Wait for the audio to finish playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

    except Exception as e:
        print(f"An error occurred during audio generation or playback: {e}")

def main():
    """
    Main function to run the voice conversation with Shrek.
    """
    # Load environment variables from a .env file
    load_dotenv()

    # --- 1. Initialize ElevenLabs Client ---
    elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
    # NOTE: VOICE_API_KEY should be the Voice ID for your desired Shrek voice in ElevenLabs
    voice_id = os.getenv("VOICE_API_KEY")

    if not elevenlabs_api_key or not voice_id:
        print("Error: Ensure 'ELEVENLABS_API_KEY' and 'VOICE_API_KEY' are set in your .env file.")
        sys.exit(1)

    try:
        elevenlabs_client = ElevenLabs(api_key=elevenlabs_api_key)
    except Exception as e:
        print(f"Error initializing ElevenLabs client: {e}")
        sys.exit(1)

    # --- 2. Setup RAG Chain for Shrek ---
    # This path is relative to your root 'ai-service' folder and should work correctly
    SCRIPTS_PATH = 'data/scripts/shrek'
    character_name = "Shrek"

    print(f"Loading dialogue for {character_name}...")
    loader = ScriptLoader(directory_path=SCRIPTS_PATH, character_name=character_name)
    docs = loader.load()

    if not docs:
        print(f"Sorry, I couldn't find any dialogue for '{character_name}'.")
        print(f"Please make sure your script files are in the '{SCRIPTS_PATH}' directory.")
        return

    # Create the conversational RAG chain
    rag_chain = create_rag_chain(docs, character_name)
    print(f"\n{character_name.capitalize()} is ready to talk! (Press Ctrl+C to quit)")

    # --- 3. Start Conversation Loop ---
    try:
        while True:
            # Get user's spoken question
            print("\nPress Enter and then speak your question for 5 seconds...")
            input() # Wait for user to press Enter to start recording
            user_question = get_transcript()
            print(f"You said: {user_question}")

            if not user_question.strip():
                print("Couldn't quite catch that. Try again.")
                continue

            # Get Shrek's text response from the RAG chain
            shrek_response_text = rag_chain.invoke(user_question)
            print(f"Shrek says: {shrek_response_text}")

            # Convert the text response to speech and play it
            generate_and_play_audio(elevenlabs_client, shrek_response_text, voice_id)

    except KeyboardInterrupt:
        print(f"\n\n{character_name.capitalize()}: 'Alright, get out of my swamp!'")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()