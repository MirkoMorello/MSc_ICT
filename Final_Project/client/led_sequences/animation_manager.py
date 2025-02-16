import time

class AnimationManager:
    def __init__(self, transition_duration=1.0):
        self.current_anim = None
        self.next_anim = None
        self.transition_start = None
        self.transition_duration = transition_duration

    def start_animation(self, new_anim):
        if self.current_anim is None:
            self.current_anim = new_anim
            new_anim.start_sequence()
        else:
            self.next_anim = new_anim
            self.transition_start = time.time()

    def update(self):
        if self.next_anim and self.transition_start:
            elapsed = time.time() - self.transition_start
            blend_factor = min(elapsed / self.transition_duration, 1.0)
            
            # Get both animations' states
            current_frame = self.current_anim.get_current_frame()
            next_frame = self.next_anim.get_current_frame()
            
            # Blend frames
            blended = []
            for c, n in zip(current_frame, next_frame):
                blended.append([
                    int(c[0] * (1 - blend_factor) + n[0] * blend_factor),
                    int(c[1] * (1 - blend_factor) + n[1] * blend_factor),
                    int(c[2] * (1 - blend_factor) + n[2] * blend_factor),
                    int(c[3] * (1 - blend_factor) + n[3] * blend_factor)
                ])
            
            self._write(blended)
            
            if blend_factor >= 1.0:
                self.current_anim.stop_sequence()
                self.current_anim = self.next_anim
                self.next_anim = None
                self.transition_start = None
        else:
            self.current_anim.update()