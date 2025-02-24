# client/utils/audio_utils.py
import numpy as np
import torch
import logging
import pyaudio # Import pyaudio
from . import logging_utils
from ..model import EncDecClassificationModel
logger = logging_utils.get_logger(__name__)


def audio_amplifier(audio_chunk, factor=1):
    """Amplifies the audio chunk by a given factor."""
    try:
        factor = float(factor)  # Ensure factor is a float
    except (ValueError, TypeError):
        logger.warning(f"Invalid amplification factor: {factor}.  Using 1.0.")
        factor = 1.0
    return (np.asarray(audio_chunk) * factor).astype(np.int16)


def get_model(path="../best_model.pth"):
    """Loads the PyTorch model.  Handles potential errors."""
    try:
        model = EncDecClassificationModel(num_classes=35) 
        model.load_state_dict(torch.load(path, map_location=torch.device('mps') if torch.mps.is_available() else torch.device('cpu')))
        model.eval()  # Set to evaluation mode
        return model
    except FileNotFoundError:
        logger.error(f"Model file not found at {path}")
        return None
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None