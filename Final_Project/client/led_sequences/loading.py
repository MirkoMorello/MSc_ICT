from led_sequences.base_sequence import BaseSequence
import time

class Loading(BaseSequence):
    def __init__(self, color, duration, brightness=0.5):
        super().__init__()
        self.color = color
        self.duration = duration
        self.brightness = brightness

    def sequence(self, semaphore):
        start_time = time.time()
        while semaphore.is_keep_going():
            current_time = time.time()
            elapsed_time = current_time - start_time
            progress = (elapsed_time % self.duration) / self.duration  # 0.0 to 1.0
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

            self._write(led_states)
            time.sleep(0.01)  # Short delay for smooth animation