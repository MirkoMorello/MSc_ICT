from led_sequences.base_sequence import BaseSequence
import colorsys
import random
import time
import math

"""

"""
class FlashRainbow(BaseSequence):
    def __init__(self, fade_duration=3.0, color_transition_duration=1.5, hue_speed=0.1):
        super().__init__()
        self.fade_duration = fade_duration
        self.color_transition_duration = color_transition_duration
        self.hue_speed = hue_speed
        self.base_hue = random.random()
        self.current_hues = [random.random() for _ in range(self.get_led_count())]
        self.target_hues = [random.random() for _ in range(self.get_led_count())]
        # Store timing state for the animation:
        self._start_time = time.time()
        self._last_transition_update = time.time()
        self._transition_start = time.time()

    def _gamma_correct(self, value):
        """Convert linear to perceptual brightness."""
        return math.pow(value, 2.2)

    def _update_targets(self):
        """Update hue targets and reset transition timing."""
        # Slowly drift the base hue:
        self.base_hue = (self.base_hue + self.hue_speed * 0.01) % 1.0
        # Update the hue transition:
        self._transition_start = time.time()
        # Current hues become the old target hues:
        self.current_hues = list(self.target_hues)
        # Generate new targets with some variation around the base hue:
        self.target_hues = [
            (self.base_hue + random.uniform(-0.1, 0.1)) % 1.0
            for _ in range(self.get_led_count())
        ]

    def get_current_frame(self):
        """
        Computes and returns the current LED frame.
        
        Each LED state is a list: [brightness_byte, blue, green, red].
        """
        current_time = time.time()
        elapsed_total = current_time - self._start_time
        fade_progress = min(elapsed_total / self.fade_duration, 1.0)
        
        # Check if it's time to update color targets:
        if (current_time - self._last_transition_update) >= self.color_transition_duration:
            self._update_targets()
            self._last_transition_update = current_time

        led_states = []
        for i in range(self.get_led_count()):
            # Compute the blend for the hue transition (using an ease-in via a sine function)
            t = (current_time - self._transition_start) / self.color_transition_duration
            blend = math.sin(t * math.pi * 0.5)
            current_hue = (self.current_hues[i] + (self.target_hues[i] - self.current_hues[i]) * blend) % 1.0
            
            # Convert HSV to RGB (values are in 0-1 range)
            r, g, b = colorsys.hsv_to_rgb(current_hue, 1, 1)
            
            # Apply gamma correction
            r = self._gamma_correct(r)
            g = self._gamma_correct(g)
            b = self._gamma_correct(b)
            
            # Blend with white based on fade_progress
            white_blend = 1 - fade_progress
            r = (r * (1 - white_blend)) + (1.0 * white_blend)
            g = (g * (1 - white_blend)) + (1.0 * white_blend)
            b = (b * (1 - white_blend)) + (1.0 * white_blend)
            
            # Convert colors to 0-255 with clamping
            r = max(0, min(255, int(r * 255)))
            g = max(0, min(255, int(g * 255)))
            b = max(0, min(255, int(b * 255)))
            
            # Compute brightness (using a smooth decay)
            brightness = 1.0 - math.pow(fade_progress, 2)
            brightness_byte = 0xE0 | int(brightness * 31)
            
            led_states.append([brightness_byte, b, g, r])
        return led_states

    def sequence(self, semaphore):
        """
        Main loop for the animation: retrieve the current frame and write it to the LED strip.
        """
        while semaphore.is_keep_going():
            frame = self.get_current_frame()
            self._write(frame)
            time.sleep(0.005)  # Short delay for smoother updates
