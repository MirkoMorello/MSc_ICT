import logging
from dotenv import load_dotenv
import os
import pyaudio

# Logging Configuration
LOGGING_LEVEL = logging.INFO

# LED Configuration
LED_COUNT = 12
START_FRAME = [0x00, 0x00, 0x00, 0x00]
END_FRAME = [0xFF, 0xFF, 0xFF, 0xFF]

# Additional Configuration for VAD
DEPLOYMENT_MODE = os.getenv("DEPLOYMENT_MODE", "dev")
AMPLIFICATION_FACTOR_VAD = 1.0  # Set to an appropriate amplification factor

SERVER_ADDRESS = os.getenv("SERVER_ADDRESS", "localhost")
SERVER_PORT = int(os.getenv("SERVER_PORT", 8080))
AUDIO_FORMAT = pyaudio.paInt16
AUDIO_CHANNELS = 1
AUDIO_RATE = 16000
AUDIO_CHUNK_SIZE = 1024
VAD_THRESHOLD = os.getenv("VAD_THRESHOLD", 60)
WAKE_WORD = os.getenv("WAKE_WORD", "marvin")
PROBABILITY_THRESHOLD = 0.25
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models", "best_model.pth")
AMPLIFICATION_FACTOR_VAD = os.getenv("AMPLIFICATION_FACTOR_VAD", 1.0)
AMPLIFICATION_FACTOR_WAKE_WORD = os.getenv("AMPLIFICATION_FACTOR_WAKE_WORD", 1.0)
LABELS = [
    'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
    'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin',
    'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six',
    'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'
]