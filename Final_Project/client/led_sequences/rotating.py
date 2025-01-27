from led_sequences.base_sequence import BaseSequence
import time

class Rotating(BaseSequence):
    def __init__(self, color, brightness = 0.5):
        super().__init__()
        self.brightness = brightness
        self.color = color
         
    def sequence(self, semaphore):
        while(semaphore.is_keep_going()):
            led_state = []
            for i in range(self.get_led_count()):
                # Get the rainbow color for this position
                r, g, b = self.color

                led_state = ([[0xE0, 0x00, 0x00, 0x00]] * i) +  [[0xE0 | int(self.brightness * 31), b, g, r]] + ([[0xE0, 0x00, 0x00, 0x00]] * (self.get_led_count() - i - 1))

                self._write(led_state)
                time.sleep(0.01)

                self.turn_off_leds()
                time.sleep(0.01)