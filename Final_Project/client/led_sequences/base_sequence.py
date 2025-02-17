from config import (
    START_FRAME, 
    END_FRAME, 
    LED_COUNT
)
import spidev
import threading
from abc import ABC, abstractmethod

"""
    This is the abstract class used for represents a led sequence.
"""
class BaseSequence(ABC):

    """
        This class is used for representing a semaphore which is used for controlling
        when an animation should terminate or not.
    """
    class Semaphore:
        def __init__(self):
            self.keep_going = False

        def is_keep_going(self):
            return self.keep_going

        def stop(self):
            self.keep_going = False
        
        def reset(self):
            self.keep_going = True

    def __init__(self, led_count=LED_COUNT):
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

    """
        Method used for writing data to the i2c interface
    """
    def _write(self, data):
        if len(data) != self.led_count:
            raise ValueError("A list of length {} is required, where each element is a list composed of BRIGHTNESS + BGR".format(self.led_count))
        led_data = []
        led_data.extend(START_FRAME)
        for led in data:
            if min(led) < 0 or max(led) > 255:
                raise ValueError("Each element in the list must be an integer between 0 and 255")
            led_data.extend(led)
        led_data.extend(END_FRAME)
        self.spi.xfer2(led_data)

    """
        Method used for starting a led sequence
    """
    def start_sequence(self):
        if not self.__semaphore.is_keep_going():
            self.__semaphore.reset()
            self.thread = threading.Thread(target=self.sequence, args=(self.__semaphore,))
            self.thread.start()
    
    """
        Method used for stopping a led sequence
    """
    def stop_sequence(self, blocking=True):
        self.__semaphore.stop()
        if blocking:
            self.thread.join()
        self.turn_off_leds()

    """
        Method used for turning all the leds off
    """
    def turn_off_leds(self):
        led_data = []
        led_data.extend(START_FRAME)
        led_data.extend([0xE0, 0x00, 0x00, 0x00] * self.led_count)
        led_data.extend(END_FRAME)
        self.spi.xfer2(led_data)

    def blend_frames(self, frame1, frame2, blend_factor):
        """
        Blends two LED frames based on the given blend factor.
        A blend_factor of 0 returns frame1; a blend_factor of 1 returns frame2.
        Both frames must be lists of length `led_count` where each element is a list of 4 ints.
        """
        if len(frame1) != self.led_count or len(frame2) != self.led_count:
            raise ValueError("Both frames must have length equal to led_count.")
        blended_frame = []
        for led1, led2 in zip(frame1, frame2):
            blended_led = [
                int(round(v1 * (1 - blend_factor) + v2 * blend_factor))
                for v1, v2 in zip(led1, led2)
            ]
            blended_frame.append(blended_led)
        return blended_frame

    @abstractmethod
    def sequence(self, semaphore):
        pass

    @abstractmethod
    def get_current_frame(self):
        pass
