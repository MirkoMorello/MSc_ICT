import threading
import time
from led_sequences.base_sequence import BaseSequence

class AnimationManager:
    def __init__(self, initial_animation: BaseSequence):
        self.current_animation = initial_animation
        self.next_animation = None
        self.transitioning = False
        self.transition_duration = 1.0  # Default transition time in seconds
        self.lock = threading.Lock()
        self.thread = None  # Track animation thread

        # Start the initial animation
        self.current_animation.start_sequence()

    def set_animation(self, new_animation: BaseSequence, transition_duration=1.0):
        """Smoothly transition from the current animation to the new one."""
        with self.lock:
            if self.transitioning:
                return  # Prevent overlapping transitions

            self.next_animation = new_animation
            self.transition_duration = transition_duration
            self.transitioning = True

            # Start the transition in a separate thread
            self.thread = threading.Thread(target=self._handle_transition)
            self.thread.start()

    def _handle_transition(self):
        """Smooth transition between animations by blending frames."""
        start_time = time.time()
        while time.time() - start_time < self.transition_duration:
            progress = (time.time() - start_time) / self.transition_duration
            progress = min(1.0, progress)  # Ensure it never goes above 1.0

            # Get frames from both animations
            old_frame = self.current_animation.get_current_frame()
            new_frame = self.next_animation.get_current_frame()

            # Blend frames
            blended_frame = self._blend_frames(old_frame, new_frame, progress)

            # Send blended frame to LEDs
            self.current_animation._write(blended_frame)
            time.sleep(0.05)  # Smooth blending

        # Complete transition
        with self.lock:
            self.current_animation.stop_sequence(blocking=False)  # Stop old animation only after transition
            self.current_animation = self.next_animation
            self.current_animation.start_sequence()
            self.next_animation = None
            self.transitioning = False

    def _blend_frames(self, old_frame, new_frame, progress):
        """Blend two frames based on transition progress."""
        blended_frame = []
        for old_led, new_led in zip(old_frame, new_frame):
            blended_led = [
                int(old_val * (1 - progress) + new_val * progress) for old_val, new_val in zip(old_led, new_led)
            ]
            blended_frame.append(blended_led)
        return blended_frame

    def stop_sequence(self, blocking=True):
        """Stop the current animation gracefully."""
        with self.lock:
            if self.current_animation:
                self.current_animation.stop_sequence(blocking)
