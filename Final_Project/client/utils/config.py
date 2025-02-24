# client/config.py
import os
from dotenv import load_dotenv
import pyaudio

load_dotenv()
# LED Configuration
LED_COUNT = 12
START_FRAME = [0x00, 0x00, 0x00, 0x00]
END_FRAME = [0xFF, 0xFF, 0xFF, 0xFF]


SERVER_ADDRESS = os.getenv("SERVER_ADDRESS", "localhost")
SERVER_PORT = int(os.getenv("SERVER_PORT", 8080))
DEPLOYMENT_MODE = os.getenv("DEPLOYMENT_MODE", "dev")
WAKE_WORD = "marvin"
PROBABILITY_THRESHOLD = 0.99
LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "DEBUG")
DEBUG = True
AUDIO_FORMAT = pyaudio.paInt16
AUDIO_CHANNELS = 1
AUDIO_RATE = 16000
AUDIO_CHUNK_SIZE = int(AUDIO_RATE * 0.5)  # 0.5 seconds
VAD_THRESHOLD = int(os.getenv("VAD_THRESHOLD", 300))  # Ensure it's an integer
AMPLIFICATION_FACTOR_VAD = float(os.getenv("AMPLIFICATION_FACTOR_VAD", 1.0))
AMPLIFICATION_FACTOR_WAKE_WORD = float(os.getenv("AMPLIFICATION_FACTOR_WAKE_WORD", 1.0))
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../models", "best_model.pth")
LABELS = [
    'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
    'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin',
    'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six',
    'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'
]
MIN_SILENT_FRAMES_SILENT = 10
MIN_SILENT_FRAMES_VOICED = 4