import wave
import numpy as np
import logging
from . import logging_utils
logger = logging_utils.get_logger(__name__)

def save_audio_to_wav(audio_data: np.ndarray, sample_rate: int, filename: str):
    """Saves audio data to a WAV file."""
    try:
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            int16_data = (audio_data * 32767).astype(np.int16)
            wf.writeframes(int16_data.tobytes())
    except Exception as e:
        logger.error(f"Error saving audio to WAV: {e}")