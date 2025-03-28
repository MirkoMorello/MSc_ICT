import os
import numpy as np
import time
import torch
from typing import Tuple, Optional
from . import logging_utils    # Changed: .utils.logging_utils
from ..config import KOKORO_LANG_CODE

try:
    from kokoro.pipeline import KPipeline
except ImportError:
    KPipeline = None  # Handle import error, this is a fallback 

logger = logging_utils.get_logger(__name__)

class TTSHandler:
    def __init__(self):
        self.tts_pipeline = self.build_kokoro_tts()

    def build_kokoro_tts(self):
        if KPipeline is None:
          logger.error("Kokoro TTS is not installed correctly.")
          return None

        try:
            tts_pipeline = KPipeline(lang_code=KOKORO_LANG_CODE)
            return tts_pipeline
        except Exception as e:
            logger.exception(f"Failed to initialize Kokoro TTS: {e}")
            return None

    def generate_tts(self, text) -> Tuple[Optional[np.ndarray], Optional[str]]:
        if self.tts_pipeline is None:
            logger.error("No Kokoro TTS pipeline available.")
            return None, None

        try:
            start_time = time.perf_counter()
            generator = self.tts_pipeline(
                text,
                voice='af_heart',
                speed=1,
                split_pattern=None
            )
            merged_audio = []
            debug_phonemes = []
            for gs, ps, chunk_audio in generator:
                merged_audio.extend(chunk_audio.tolist())
                debug_phonemes.append(ps)

            merged_audio = np.array(merged_audio, dtype=np.float32)

            # --- ADD THESE CHECKS ---
            merged_audio = np.nan_to_num(merged_audio, nan=0.0, posinf=1.0, neginf=-1.0)  # Replace
            if not np.any(merged_audio):  # Check if all zeros
                logger.warning("TTS generated all-zero audio.")
                return None, None
            # --- END CHECKS ---

            audio_norm = merged_audio / np.max(np.abs(merged_audio))
            audio_norm = (audio_norm * 32767).astype(np.int16)

            elapsed = (time.perf_counter() - start_time) * 1000
            logger.info(f"🟠 TTS finished in {elapsed:.2f} ms. Audio length={len(merged_audio)} samples.")
            phoneme_str = " | ".join(debug_phonemes) if debug_phonemes else None
            return audio_norm, phoneme_str

        except Exception as e:
            logger.exception(f"TTS generation error: {e}")
            return None, None