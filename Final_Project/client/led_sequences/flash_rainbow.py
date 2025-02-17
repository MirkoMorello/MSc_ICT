from led_sequences.base_sequence import BaseSequence
import colorsys
import random
import time
import math

class FlashRainbow(BaseSequence):
    def __init__(self, fade_duration=3.0, color_transition_duration=1.5, hue_speed=0.1):
        super().__init__()
        self.fade_duration = fade_duration
        self.color_transition_duration = color_transition_duration
        self.hue_speed = hue_speed
        self.base_hue = random.random()
        self.current_hues = [random.random() for _ in range(self.get_led_count())]
        self.target_hues = [random.random() for _ in range(self.get_led_count())]
        self._start_time = time.time()
        self._last_transition_update = time.time()
        self._transition_start = time.time()

    def _gamma_correct(self, value):
        return math.pow(value, 2.2)

    def _update_targets(self):
        self.base_hue = (self.base_hue + self.hue_speed * 0.01) % 1.0
        self._transition_start = time.time()
        self.current_hues = list(self.target_hues)
        self.target_hues = [
            (self.base_hue + random.uniform(-0.1, 0.1)) % 1.0
            for _ in range(self.get_led_count())
        ]

    def get_current_frame(self):
        current_time = time.time()
        elapsed_total = current_time - self._start_time
        fade_progress = min(elapsed_total / self.fade_duration, 1.0)
        if (current_time - self._last_transition_update) >= self.color_transition_duration:
            self._update_targets()
            self._last_transition_update = current_time
        led_states = []
        for i in range(self.get_led_count()):
            t = (current_time - self._transition_start) / self.color_transition_duration
            blend = math.sin(t * math.pi * 0.5)
            current_hue = (self.current_hues[i] + (self.target_hues[i] - self.current_hues[i]) * blend) % 1.0
            r, g, b = colorsys.hsv_to_rgb(current_hue, 1, 1)
            r = self._gamma_correct(r)
            g = self._gamma_correct(g)
            b = self._gamma_correct(b)
            white_blend = 1 - fade_progress
            r = (r * (1 - white_blend)) + (1.0 * white_blend)
            g = (g * (1 - white_blend)) + (1.0 * white_blend)
            b = (b * (1 - white_blend)) + (1.0 * white_blend)
            r = max(0, min(255, int(r * 255)))
            g = max(0, min(255, int(g * 255)))
            b = max(0, min(255, int(b * 255)))
            brightness = 1.0 - math.pow(fade_progress, 2)
            brightness_byte = 0xE0 | int(brightness * 31)
            led_states.append([brightness_byte, b, g, r])
        return led_states

    def sequence(self, semaphore):
        while semaphore.is_keep_going():
            frame = self.get_current_frame()
            self._write(frame)
            time.sleep(0.005)
