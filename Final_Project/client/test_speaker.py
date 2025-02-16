from pydub import AudioSegment
from pydub.playback import play

AUDIO_FILE = "test_recording.wav"
BOOST_DB = 30  # Increase volume by 30 dB (Adjust as needed)

try:
    # Load the audio file
    sound = AudioSegment.from_wav(AUDIO_FILE)

    # Increase volume
    louder_sound = sound + BOOST_DB  # Increase dB
    print(f"Playing {AUDIO_FILE} with +{BOOST_DB} dB volume boost...")

    # Play the boosted audio
    play(louder_sound)

    print("Audio playback finished.")

except FileNotFoundError:
    print(f"Error: File '{AUDIO_FILE}' not found.")
except Exception as e:
    print(f"Error playing file: {e}")
