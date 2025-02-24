import pyaudio
from .client import Client  # Use relative import: .client
from .utils.audio_utils import get_model  # Use relative import: .utils.audio_utils
from .utils.config import (  # Use relative import: .config
    SERVER_ADDRESS, SERVER_PORT, AUDIO_FORMAT, AUDIO_CHANNELS, AUDIO_RATE,
    AUDIO_CHUNK_SIZE, VAD_THRESHOLD, MODEL_PATH, LABELS
)

from .utils import logging_utils  # Use relative import
logger = logging_utils.get_logger(__name__)


if __name__ == "__main__":
    audio_params = {
        "format": AUDIO_FORMAT,
        "channels": AUDIO_CHANNELS,
        "rate": AUDIO_RATE,
        "chunk_size": AUDIO_CHUNK_SIZE,
        "threshold": VAD_THRESHOLD
    }

    model = get_model(path = MODEL_PATH)
    if model is None:
        logger.error("Error: Model loading failed.")
        exit()


    client = Client(
        server_ip=SERVER_ADDRESS,
        server_port=SERVER_PORT,
        audio_params=audio_params,
        model=model,
        labels=LABELS,
    )
    client.run()