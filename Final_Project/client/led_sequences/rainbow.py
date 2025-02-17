from led_sequences.base_sequence import BaseSequence
import time 

"""
    This class represents a rainbow animation characterized by all the leds assuming a different rainbow color in a circular movement. 
"""
class Rainbow(BaseSequence):
    def __init__(self, brightness=0.5):
        super().__init__()
        self.brightness = brightness
        self.start_time = time.time()

    def __wheel(self, pos):
        """Generate rainbow colors across 0-255 positions."""
        if pos < 85:
            return (pos * 3, 255 - pos * 3, 0)
        elif pos < 170:
            pos -= 85
            return (255 - pos * 3, 0, pos * 3)
        else:
            pos -= 170
            return (0, pos * 3, 255 - pos * 3)

    def get_current_frame(self):
        """
        Computes and returns the current LED frame for the rainbow animation.
        """
        elapsed = int((time.time() - self.start_time) * 100) % 256
        led_data = []

        for i in range(self.get_led_count()):
            color = self.__wheel((int(i * 256 / self.get_led_count()) + elapsed) & 255)
            r, g, b = [int(c * self.brightness) for c in color]
            led_data.append([0xE0 | int(self.brightness * 31), b, g, r])
        
        return led_data

    def sequence(self, semaphore):
        """Draw a rainbow that smoothly cycles across all LEDs."""
        while semaphore.is_keep_going():
            frame = self.get_current_frame()
            self._write(frame)
            time.sleep(0.01)
