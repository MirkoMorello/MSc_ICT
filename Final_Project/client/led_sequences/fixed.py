from led_sequences.base_sequence import BaseSequence
import time

class Fixed(BaseSequence):
    def __init__(self, color, brightness = 0.5):
        super().__init__()
        self.color = color
        self.brightness = brightness

    def sequence(self, semaphore):
        r, g, b = self.color
        max_brightness = int(self.brightness * 31)

        while(semaphore.is_keep_going()):
            led_data = []
            for _ in range(self.get_led_count()):
                led_data.append([0xE0 | max_brightness, b, g, r])
            self._write(led_data)
            time.sleep(0.01)
    