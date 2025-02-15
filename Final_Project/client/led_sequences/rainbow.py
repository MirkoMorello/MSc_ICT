from led_sequences.base_sequence import BaseSequence
import time 

class Rainbow(BaseSequence):
    def __init__(self, brightness = 0.5):
        super().__init__()
        self.brightness = brightness

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
        
    def sequence(self, semaphore):
        """Draw rainbow that uniformly distributes itself across all pixels."""
        while semaphore.is_keep_going():
            for j in range(256):
                if not semaphore.is_keep_going(): 
                    break
                led_data = []
                for i in range(self.get_led_count()):
                    color = self.__wheel((int(i * 256 / self.get_led_count()) + j) & 255)
                    r, g, b = [int(c * self.brightness) for c in color]
                    led_data.append([0xE0 | int(self.brightness*31), b, g, r]) # Brightness + BGR
                self._write(led_data)
                time.sleep(0.01)
