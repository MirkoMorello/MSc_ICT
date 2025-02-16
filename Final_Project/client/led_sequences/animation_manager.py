import threading
import time

class AnimationManager:
    def __init__(self, initial_animation, update_interval=0.01):
        self.current_animation = initial_animation
        self.next_animation = None
        self.transition_duration = 1.0
        self.transition_start = None
        self.transitioning = False
        self.update_interval = update_interval
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def effective_animation(self):
        """Return the animation that will eventually be active.
        
        If a transition is in progress, this returns the target animation;
        otherwise, it returns the current animation.
        """
        with self.lock:
            return self.next_animation if self.transitioning else self.current_animation

    def _run(self):
        while self.running:
            with self.lock:
                if self.transition_start is not None:
                    elapsed = time.time() - self.transition_start
                    progress = min(elapsed / self.transition_duration, 1.0)
                    current_frame = self.current_animation.get_current_frame()
                    next_frame = self.next_animation.get_current_frame()
                    frame_to_write = self._blend_frames(current_frame, next_frame, progress)
                    if progress >= 1.0:
                        self.current_animation = self.next_animation
                        self.next_animation = None
                        self.transition_start = None
                        self.transitioning = False
                else:
                    frame_to_write = self.current_animation.get_current_frame()
            self.current_animation._write(frame_to_write)
            time.sleep(self.update_interval)

    def _blend_frames(self, frame1, frame2, progress):
        blended_frame = []
        for led1, led2 in zip(frame1, frame2):
            # Blend brightness channel separately (LED format: [brightness, blue, green, red])
            brightness1 = led1[0] & 0x1F
            brightness2 = led2[0] & 0x1F
            blended_brightness = int(brightness1 * (1 - progress) + brightness2 * progress)
            brightness_byte = 0xE0 | blended_brightness

            blue = int(led1[1] * (1 - progress) + led2[1] * progress)
            green = int(led1[2] * (1 - progress) + led2[2] * progress)
            red = int(led1[3] * (1 - progress) + led2[3] * progress)
            blended_frame.append([brightness_byte, blue, green, red])
        return blended_frame

    def set_animation(self, new_animation, transition_duration=1.0):
        with self.lock:
            # (Optionally, you could also decide to ignore new transitions if one is in progress.)
            self.next_animation = new_animation
            self.transition_duration = transition_duration
            self.transition_start = time.time()
            self.transitioning = True

    def change_animation_if_needed(self, desired_type, new_animation, transition_duration=1.0):
        """
        Change the animation only if there is no transition in progress
        and the effective animation isn't already of the desired type.
        """
        with self.lock:
            if self.transitioning:
                return  # A transition is already in progress.
            if not isinstance(self.effective_animation(), desired_type):
                self.set_animation(new_animation, transition_duration)

    def stop(self):
        self.running = False
        self.thread.join()
        self.current_animation.turn_off_leds()


