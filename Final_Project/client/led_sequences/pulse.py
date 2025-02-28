from .base_sequence import BaseSequence
import time

"""
    This class represents an animation consisting in all the led pulsing with a certain frequency and color
"""
class Pulse(BaseSequence):
    def __init__(self, color, brightness=0.5):
        super().__init__()
        self.color = color
        self.brightness = brightness
        self.start_time = time.time()

    def get_current_frame(self):
        """
        Computes and returns the current LED frame based on pulse animation.
        """
        r, g, b = self.color
        max_brightness = int(self.brightness * 31)

        # Determine current brightness level
        elapsed = (time.time() - self.start_time) % 2  # 2s cycle: 1s fade in, 1s fade out
        if elapsed < 1:
            brightness = int(max_brightness * elapsed)  # Fade in
        else:
            brightness = int(max_brightness * (2 - elapsed))  # Fade out

        # Create LED frame
        led_data = [[0xE0 | brightness, b, g, r] for _ in range(self.get_led_count())]
        return led_data

    def sequence(self, semaphore):
        while semaphore.is_keep_going():
            frame = self.get_current_frame()
            self._write(frame)
            time.sleep(0.01) 
