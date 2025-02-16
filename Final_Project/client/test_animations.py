from led_sequences.animation_manager import AnimationManager
from led_sequences.fixed import Fixed
from led_sequences.pulse import Pulse
from led_sequences.loading import Loading
import time

fixed_anim = Fixed(color=(0, 0, 0), brightness=0.5)

anim_manager = AnimationManager(fixed_anim)
#time.sleep(5) 
pulse_anim = Pulse(color=(0, 255, 0), brightness=0.7)
anim_manager.set_animation(pulse_anim, transition_duration=0.5)
time.sleep(5)

loading = Loading(color=(0, 0, 255), brightness=0.7, duration=5)
anim_manager.set_animation(loading, transition_duration=1.5)
time.sleep(10)

# When you’re done (e.g., on shutdown)
anim_manager.stop()
