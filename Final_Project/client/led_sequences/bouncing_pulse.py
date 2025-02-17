from led_sequences.base_sequence import BaseSequence
import time
import math

class BouncingPulse(BaseSequence):
    def __init__(self, color, duration, wave_count=2, smoothness=2):
        super().__init__()
        self.color = color
        self.duration = duration
        self.wave_count = wave_count  # Number of waves per cycle
        self.smoothness = smoothness  # Easing smoothness (1-4)
        self.start_time = time.time()  # Store start time for consistent timing

    def _wave_function(self, t):
        """Composite wave function with sinusoidal easing."""
        # Primary sine wave with adjustable wave count
        primary = math.sin(math.pi * self.wave_count * t)
        # Smoothing function using power easing
        eased = math.pow(primary, self.smoothness)
        # Normalize and clamp values
        return max(0.0, min(1.0, abs(eased)))

    def get_current_frame(self):
        """
        Computes and returns the current LED frame.
        Each LED state is a list: [brightness_byte, blue, green, red].
        """
        current_time = time.time()
        elapsed = current_time - self.start_time
        led_count = self.get_led_count()
        phase_offset = 0.1  # Small phase offset between LEDs for extra smoothness
        
        base_progress = (elapsed % self.duration) / self.duration
        
        led_states = []
        for i in range(led_count):
            # Add slight phase variation between LEDs
            led_progress = (base_progress + (i * phase_offset / led_count)) % 1.0
            # Calculate wave intensity
            intensity = self._wave_function(led_progress)
            # Apply exponential brightness scaling for perceptually linear fade
            brightness = math.pow(intensity, 1.5)
            
            # Calculate color components based on brightness
            r, g, b = self.color
            r = max(0, min(255, int(r * brightness)))
            g = max(0, min(255, int(g * brightness)))
            b = max(0, min(255, int(b * brightness)))
            
            # Build LED command: 0xE0 | (brightness scaled to 5 bits)
            brightness_byte = 0xE0 | int(brightness * 31)
            led_states.append([brightness_byte, b, g, r])
            
        return led_states

    def sequence(self, semaphore):
        while semaphore.is_keep_going():
            frame = self.get_current_frame()
            self._write(frame)
            time.sleep(0.005)  # Shorter sleep for smoother animation
