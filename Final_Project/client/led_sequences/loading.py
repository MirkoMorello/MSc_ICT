from .base_sequence import BaseSequence
import time

"""
    This class represents a loading animation showed during the grace period before the data are sent to the server
"""
class Loading(BaseSequence):
    def __init__(self, color, duration, brightness=0.5):
        super().__init__()
        self.color = color
        self.duration = duration
        self.brightness = brightness
        self.start_time = time.time()

    def get_current_frame(self):
        """
        Computes and returns the current LED frame.
        Each LED state is a list: [brightness_byte, blue, green, red].
        """
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        progress = (elapsed_time % self.duration) / self.duration 

        led_count = self.get_led_count()
        led_progress = progress * led_count
        full_led = int(led_progress) 
        fraction = led_progress - full_led 

        led_states = []
        for i in range(led_count):
            if i < full_led:
                led_brightness = 1.0 
            elif i == full_led and i < led_count:
                led_brightness = fraction
            else:
                led_brightness = 0.0

            # Calculate total brightness and clamp between 0 and 1
            total_brightness = max(0.0, min(1.0, self.brightness * led_brightness))
            brightness_5bit = int(total_brightness * 31)
            brightness_byte = 0xE0 | brightness_5bit

            r, g, b = self.color
            led_states.append([brightness_byte, b, g, r])

        return led_states

    def sequence(self, semaphore):
        while semaphore.is_keep_going():
            frame = self.get_current_frame()
            self._write(frame)
            time.sleep(0.01) 
