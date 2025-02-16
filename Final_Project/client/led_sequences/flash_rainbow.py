from led_sequences.base_sequence import BaseSequence
import colorsys
import random
import time

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
        self.hue_speed = hue_speed  # 0.0-1.0 controls color change speed
        self.base_hue = random.random()
        self.current_hues = [random.random() for _ in range(self.get_led_count())]
        self.target_hues = [random.random() for _ in range(self.get_led_count())]
        self.transition_start = time.time()

    def _gamma_correct(self, value):
        """Convert linear to perceptual brightness"""
        return math.pow(value, 2.2)

    def _get_current_color(self, index):
        """Smoothly transition between hues using sinusoidal easing"""
        t = (time.time() - self.transition_start) / self.color_transition_duration
        blend = math.sin(t * math.pi * 0.5)  # Ease-in easing function
        
        current_hue = (self.current_hues[index] + 
                      (self.target_hues[index] - self.current_hues[index]) * blend)
        return colorsys.hsv_to_rgb(current_hue % 1.0, 1, 1)

    def _update_targets(self):
        # Slowly drift base hue
        self.base_hue = (self.base_hue + self.hue_speed * 0.01) % 1.0
        
        # Generate new targets with controlled variation
        self.transition_start = time.time()
        self.current_hues = [self.target_hues[i] for i in range(self.get_led_count())]
        self.target_hues = [
            (self.base_hue + random.uniform(-0.1, 0.1)) % 1.0
            for _ in range(self.get_led_count())
        ]

    def sequence(self, semaphore):
        start_time = time.time()
        last_update = time.time()
        
        while semaphore.is_keep_going():
            # Update timing values
            elapsed_total = time.time() - start_time
            elapsed_transition = time.time() - last_update
            fade_progress = min(elapsed_total / self.fade_duration, 1.0)
            
            # Update color targets periodically
            if elapsed_transition >= self.color_transition_duration:
                self._update_targets()
                last_update = time.time()

            led_states = []
            for i in range(self.get_led_count()):
                # Get current color with gamma correction
                r, g, b = self._get_current_color(i)
                r = self._gamma_correct(r)
                g = self._gamma_correct(g)
                b = self._gamma_correct(b)
                
                # Blend with white flash
                white_blend = 1 - fade_progress
                r = (r * (1 - white_blend)) + (1.0 * white_blend)
                g = (g * (1 - white_blend)) + (1.0 * white_blend)
                b = (b * (1 - white_blend)) + (1.0 * white_blend)
                
                # Convert to 0-255 with clamping
                r = max(0, min(255, int(r * 255)))
                g = max(0, min(255, int(g * 255)))
                b = max(0, min(255, int(b * 255)))
                
                # Smooth brightness decay
                brightness = 1.0 - math.pow(fade_progress, 2)
                brightness_byte = 0xE0 | int(brightness * 31)
                
                led_states.append([brightness_byte, b, g, r])

            self._write(led_states)
            time.sleep(0.005)  # Faster updates for smoother transitions