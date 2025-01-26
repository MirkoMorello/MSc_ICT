from led_sequences.base_sequence_rotation import BaseSequenceRotation
import time 

class Rainbow(BaseSequenceRotation):
    def __init__(self, brightness = 0.5):
        super().__init__()
        self.brightness = brightness

    def sequence(self, semaphore):
        """Draw rainbow that uniformly distributes itself across all pixels."""
        while semaphore.is_keep_going():
            for j in range(256):
                led_data = []
                for i in range(self.get_led_count()):
                    color = self._wheel((int(i * 256 / self.get_led_count()) + j) & 255)
                    r, g, b = [int(c * self.brightness) for c in color]
                    led_data.append([0xE0 | int(self.brightness*31), b, g, r]) # Brightness + BGR
                self._write(led_data)
                time.sleep(0.01)
