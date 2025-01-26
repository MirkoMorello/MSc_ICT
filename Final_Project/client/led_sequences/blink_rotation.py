from led_sequences.base_sequence_rotation import BaseSequenceRotation
import time

class BlinkRotation(BaseSequenceRotation):
    def __init__(self, brightness = 0.5):
        super().__init__()
        self.brightness = brightness
         
    def sequence(self, semaphore):
        while(semaphore.is_keep_going()):
            led_state = []
            for i in range(self.get_led_count()):
                # Get the rainbow color for this position
                color = self._wheel(int(i * 256 / self.get_led_count()) & 255)
                r, g, b = color

                led_state = ([[0xE0, 0x00, 0x00, 0x00]] * i) +  [[0xE0 | int(self.brightness * 31), b, g, r]] + ([[0xE0, 0x00, 0x00, 0x00]] * (self.get_led_count() - i - 1))

                self._write(led_state)
                time.sleep(0.01)

                self.turn_off_leds()
                time.sleep(0.01)