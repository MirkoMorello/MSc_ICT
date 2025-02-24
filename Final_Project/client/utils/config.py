# client/config.py
import os
from dotenv import load_dotenv

load_dotenv()

SERVER_ADDRESS = os.getenv("SERVER_ADDRESS", "localhost")
SERVER_PORT = int(os.getenv("SERVER_PORT", 8080))
DEPLOYMENT_MODE = os.getenv("DEPLOYMENT_MODE", "dev")
WAKE_WORD = "marvin"
PROBABILITY_THRESHOLD = 0.99
DEBUG = True
AUDIO_FORMAT = pyaudio.paInt16
AUDIO_CHANNELS = 1
AUDIO_RATE = 16000
AUDIO_CHUNK_SIZE = int(AUDIO_RATE * 0.5)  # 0.5 seconds
VAD_THRESHOLD = int(os.getenv("VAD_THRESHOLD", 300))  # Ensure it's an integer
AMPLIFICATION_FACTOR_VAD = float(os.getenv("AMPLIFICATION_FACTOR_VAD", 1.0))
AMPLIFICATION_FACTOR_WAKE_WORD = float(os.getenv("AMPLIFICATION_FACTOR_WAKE_WORD", 1.0))
MODEL_PATH = "../best_model.pth"  # Relative path to the model
LABELS = [
    'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
    'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin',
    'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six',
    'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'
]