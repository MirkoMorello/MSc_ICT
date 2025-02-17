from led_sequences.base_sequence import BaseSequence
import time

class Loading(BaseSequence):
    def __init__(self, color, duration, brightness=0.5):
        super().__init__()
        self.color = color
        self.duration = duration
        self.brightness = brightness
        self.start_time = time.time()  # Store start time for consistent timing

    def get_current_frame(self):
        """
        Computes and returns the current LED frame.
        Each LED state is a list: [brightness_byte, blue, green, red].
        """
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        progress = (elapsed_time % self.duration) / self.duration  # Normalized progress (0.0 to 1.0)

        led_count = self.get_led_count()
        led_progress = progress * led_count
        full_led = int(led_progress)  # Fully lit LEDs
        fraction = led_progress - full_led  # Partially lit LED progress

        led_states = []
        for i in range(led_count):
            if i < full_led:
                led_brightness = 1.0  # Fully lit
            elif i == full_led and i < led_count:
                led_brightness = fraction  # Partial brightness
            else:
                led_brightness = 0.0  # Off

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
            time.sleep(0.01)  # Short delay for smooth animation
