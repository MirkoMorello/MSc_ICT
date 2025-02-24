# client/vad.py
import numpy as np
import logging
from . import logging_utils
from .config import DEPLOYMENT_MODE, AMPLIFICATION_FACTOR_VAD
from .audio_utils import audio_amplifier
from .led_utils import set_led_animation, LED_AVAILABLE, is_led_animation

logger = logging_utils.get_logger(__name__)

class VoiceActivityDetector:
    def __init__(self, frame_duration_ms=20, threshold=60, smoothing_factor=0.8):
        self.frame_duration_ms = frame_duration_ms
        self.threshold = threshold
        self.smoothing_factor = smoothing_factor
        self.audio_buffer = []
        self.num_silent_frames = 0
        self.sample_rate = None
        self.smoothed_energy = 0
        self.voiced_frames_detected = False

    def set_sample_rate(self, sample_rate):
        self.sample_rate = sample_rate
        logger.debug(f"Sample rate set to {sample_rate}")

    def process_audio_frame(self, frame):
        if self.sample_rate is None:
            raise ValueError("Sample rate not set. Call set_sample_rate().")

        energy = np.mean(np.abs(frame))
        self.smoothed_energy = (
            self.smoothing_factor * self.smoothed_energy
            + (1 - self.smoothing_factor) * energy
        )
        logger.debug(f"Frame energy: {energy:.2f}, Smoothed energy: {self.smoothed_energy:.2f}")

        if self.smoothed_energy > self.threshold:
            self.num_silent_frames = 0
            self.voiced_frames_detected = True
            logger.debug("Voiced frame detected")
            # e.g., update LED animation here if needed
        else:
            self.num_silent_frames += 1
            logger.debug(f"Silent frame count: {self.num_silent_frames}")

        self.audio_buffer.append(frame)

        # Decide when speech ended:
        if self.voiced_frames_detected:
            is_speech_ended = self.num_silent_frames >= 7
        else:
            is_speech_ended = self.num_silent_frames >= 18

        if is_speech_ended:
            logger.info("Speech ended detected by VAD")
            audio_to_send = self.audio_buffer
            had_voiced = self.voiced_frames_detected
            # Reset state for next utterance
            self.audio_buffer = []
            self.num_silent_frames = 0
            self.voiced_frames_detected = False
            return False, audio_to_send, had_voiced
        else:
            return True, [], None