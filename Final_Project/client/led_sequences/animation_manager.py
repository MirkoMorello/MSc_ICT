import threading
import time

class AnimationManager:
    def __init__(self, initial_animation, update_interval=0.01):
        """
        initial_animation: an instance of BaseSequence (or a subclass)
        update_interval: how often (in seconds) to update the LED strip
        """
        self.current_animation = initial_animation
        self.next_animation = None
        self.transition_duration = 1.0
        self.transition_start = None  # When the transition started (None if not transitioning)
        self.update_interval = update_interval
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        """Main loop: poll the current animation(s) and write frames to the LED strip."""
        while self.running:
            with self.lock:
                if self.transition_start is not None:
                    # In transition mode: compute progress [0,1]
                    elapsed = time.time() - self.transition_start
                    progress = min(elapsed / self.transition_duration, 1.0)
                    # Get frames from both animations
                    current_frame = self.current_animation.get_current_frame()
                    next_frame = self.next_animation.get_current_frame()
                    # Blend the two frames
                    frame_to_write = self._blend_frames(current_frame, next_frame, progress)
                    if progress >= 1.0:
                        # Transition finished: new animation becomes current.
                        self.current_animation = self.next_animation
                        self.next_animation = None
                        self.transition_start = None
                else:
                    # No transition: simply get the current frame.
                    frame_to_write = self.current_animation.get_current_frame()

            # Write the frame (using whichever animation’s _write works on the LED strip)
            self.current_animation._write(frame_to_write)
            time.sleep(self.update_interval)

    def _blend_frames(self, frame1, frame2, progress):
        """
        Blend two frames so that the LED color transitions smoothly.
        Each LED is represented as [brightness_byte, blue, green, red].
        """
        blended_frame = []
        for led1, led2 in zip(frame1, frame2):
            # Extract brightness values (0-31)
            brightness1 = led1[0] & 0x1F
            brightness2 = led2[0] & 0x1F
            # Blend brightness
            blended_brightness = int(brightness1 * (1 - progress) + brightness2 * progress)
            brightness_byte = 0xE0 | blended_brightness

            # Blend each color channel (B, G, R)
            blue = int(led1[1] * (1 - progress) + led2[1] * progress)
            green = int(led1[2] * (1 - progress) + led2[2] * progress)
            red = int(led1[3] * (1 - progress) + led2[3] * progress)

            blended_frame.append([brightness_byte, blue, green, red])
        return blended_frame


    def set_animation(self, new_animation, transition_duration=1.0):
        """
        Begin a smooth transition to a new animation.
        new_animation: a BaseSequence instance.
        transition_duration: duration of the blend (in seconds).
        """
        with self.lock:
            self.next_animation = new_animation
            self.transition_duration = transition_duration
            self.transition_start = time.time()

    def stop(self):
        """Stop the manager loop and turn off LEDs."""
        self.running = False
        self.thread.join()
        # Optionally turn off LEDs after stopping:
        self.current_animation.turn_off_leds()
