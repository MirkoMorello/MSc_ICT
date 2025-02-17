from led_sequences.base_sequence import BaseSequence
import time

"""
    This class represents a rotating animation which consists in a single led of a given color rotating around a circular pattern.
"""
class Rotating(BaseSequence):
    def __init__(self, color, brightness=0.5, speed=0.5):
        super().__init__()
        if not 0 <= brightness <= 1: 
            raise ValueError("Brightness should be between 0 and 1 (included)")
        if not 0 <= speed <= 1:
            raise ValueError("Speed should be between 0 and 1 (included)")

        self.brightness = brightness
        self.color = color
        self.speed = 0.7 * speed + 0.1
        self.current_index = 0  # Track the current LED position
        self.start_time = time.time()

    def get_current_frame(self):
        """Returns the current LED state of the rotating animation."""
        led_state = [[0xE0, 0x00, 0x00, 0x00]] * self.get_led_count()  # Initialize LEDs as off
        r, g, b = self.color
        brightness_byte = 0xE0 | int(self.brightness * 31)

        # Calculate the current position based on elapsed time
        elapsed = time.time() - self.start_time
        self.current_index = int((elapsed / self.speed) % self.get_led_count())

        # Set the active LED
        led_state[self.current_index] = [brightness_byte, b, g, r]

        return led_state

    def sequence(self, semaphore):
        """Runs the rotating light animation."""
        while semaphore.is_keep_going():
            frame = self.get_current_frame()
            self._write(frame)
            time.sleep(self.speed)
            self.turn_off_leds()
