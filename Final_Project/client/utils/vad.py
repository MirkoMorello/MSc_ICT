# client/vad.py
import numpy as np
import logging
from . import logging_utils
from ..config import DEPLOYMENT_MODE, AMPLIFICATION_FACTOR_VAD
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

    def process_audio_frame(self, frame):
        if self.sample_rate is None:
            raise ValueError("Sample rate not set. Call set_sample_rate().")

        energy = np.mean(np.abs(frame))

        self.smoothed_energy = (
            self.smoothing_factor * self.smoothed_energy
            + (1 - self.smoothing_factor) * energy
        )

        if self.smoothed_energy > self.threshold:
            self.num_silent_frames = 0
            self.voiced_frames_detected = True
            if LED_AVAILABLE and not is_led_animation("rainbow"):
                set_led_animation("rainbow", transition_duration=0.2)

        else:
            self.num_silent_frames += 1
            if LED_AVAILABLE and not is_led_animation("loading"):
                set_led_animation("loading", transition_duration=0.2)


        if False:
            logger.debug(f"Energy: {energy}, Smoothed Energy: {self.smoothed_energy:.2f}, Silent: {self.num_silent_frames}")

        self.audio_buffer.append(frame)

        if self.voiced_frames_detected:
            is_speech_ended = self.num_silent_frames >= 7
        else:
            is_speech_ended = self.num_silent_frames >= 18

        if is_speech_ended:
            logger.debug("Speech ended")
            audio_to_send = self.audio_buffer
            had_voiced = self.voiced_frames_detected
            self.audio_buffer = []
            self.num_silent_frames = 0
            self.voiced_frames_detected = False
            return False, audio_to_send, had_voiced
        else:
            return True, [], None