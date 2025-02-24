# config.py
import os
from dotenv import load_dotenv
import logging

load_dotenv()

SERVER_IP = "0.0.0.0"
SERVER_PORT = 8080
SAMPLE_RATE = 16000
HF_TOKEN = os.getenv("HF_AUTH_TOKEN", None)
KOKORO_LANG_CODE = os.getenv("KOKORO_LANG_CODE", "a")
LLM_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
LOGGING_LEVEL = logging.DEBUG  # Or logging.INFO, logging.WARNING, etc.
SPEAKER_EMBEDDINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils", "speaker_embeddings.json")
