from config import (
    START_FRAME, 
    END_FRAME, 
    LED_COUNT
)
import spidev
import threading
from abc import ABC, abstractmethod

class BaseSequence(ABC):
    class Semaphore:
        def __init__(self):
            self.keep_going = False

        def is_keep_going(self):
            return self.keep_going

        def stop(self):
            self.keep_going = False
        
        def reset(self):
            self.keep_going = True

    def __init__(self, led_count = LED_COUNT):
        self.led_count = led_count

        self.spi = spidev.SpiDev()
        self.spi.open(0, 0)
        self.spi.max_speed_hz = 8000000

        self.__semaphore = self.Semaphore()

    def __del__(self):
        self.turn_off_leds()
        self.spi.close()
    
    def get_led_count(self):
        return self.led_count

    def _write(self, data):
        if (len(data) != self.led_count):
            raise ValueError("A list of length {} is required, where each element is a list composed of BRIGHTNESS + BGR".format(self.led_count))
        led_data = []
        led_data.extend(START_FRAME)
        for led in data:
            if min(led) < 0 or max(led) > 255:
                raise ValueError("Each element in the list must be an integer between 0 and 255")
            led_data.extend(led) 
        led_data.extend(END_FRAME)
        self.spi.xfer2(led_data)

    def start_sequence(self):
        if not self.__semaphore.is_keep_going():
            self.__semaphore.reset()
            self.thread = threading.Thread(target=self.sequence, args=(self.__semaphore,))
            self.thread.start()
    
    def stop_sequence(self, blocking=True):
        self.__semaphore.stop()
        if blocking:
            self.thread.join()
        self.turn_off_leds()

    def turn_off_leds(self):
        led_data = []
        led_data.extend(START_FRAME)
        led_data.extend([0xE0, 0x00, 0x00, 0x00] * self.led_count)
        led_data.extend(END_FRAME)
        self.spi.xfer2(led_data)

    @abstractmethod
    def sequence(self, semaphore):
        pass

    @abstractmethod
    def get_current_frame(self):
        pass