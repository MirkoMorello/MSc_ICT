from led_sequences.base_sequence import BaseSequence
import time

class Rotating(BaseSequence):
    def __init__(self, color, brightness = 0.5, speed = 0.5):
        super().__init__()
        if not 0<=brightness<=1: 
            raise ValueError("Brightness should be between 0 and 1 (included)")
        if not 0<=speed<=1:
            raise ValueError("Speed should be between 0 and 1 (included)")
        self.brightness = brightness
        self.color = color
        self.speed = 0.7 * speed + 0.1
         
    def sequence(self, semaphore):
        while(semaphore.is_keep_going()):
            led_state = []
            for i in range(self.get_led_count()):
                # Get the rainbow color for this position
                r, g, b = self.color

                led_state = ([[0xE0, 0x00, 0x00, 0x00]] * i) +  [[0xE0 | int(self.brightness * 31), b, g, r]] + ([[0xE0, 0x00, 0x00, 0x00]] * (self.get_led_count() - i - 1))

                self._write(led_state)
                time.sleep(self.speed)
                self.turn_off_leds()