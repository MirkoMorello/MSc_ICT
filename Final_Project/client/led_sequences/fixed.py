from .base_sequence import BaseSequence
import time

"""
    This class represents an led animation which consists in all the leds turning on and keepking that state until the stop_sequence method is not called
"""
class Fixed(BaseSequence):
    def __init__(self, color, brightness=0.5):
        super().__init__()
        self.color = color
        self.brightness = brightness

    def get_current_frame(self):
        """
        Returns the current LED frame as a list of LED states.
        Each LED state is a list: [brightness_byte, blue, green, red].
        """
        r, g, b = self.color
        max_brightness = int(self.brightness * 31)
        frame = []
        for _ in range(self.get_led_count()):
            frame.append([0xE0 | max_brightness, b, g, r])
        return frame

    def sequence(self, semaphore):
        while semaphore.is_keep_going():
            frame = self.get_current_frame()
            self._write(frame)
            time.sleep(0.01)
