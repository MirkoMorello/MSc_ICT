# client/utils/led_utils.py (Optional - Only if you have LEDs)
import os
import logging
from . import logging_utils
from .config import DEPLOYMENT_MODE

logger = logging_utils.get_logger(__name__)

LED_AVAILABLE = False

if DEPLOYMENT_MODE == "prod":
    try:
        from led_sequences.fixed import Fixed
        from led_sequences.rainbow import Rainbow
        from led_sequences.colors import Colors
        from led_sequences.animation_manager import AnimationManager
        from led_sequences.pulse import Pulse

        rainbow = Rainbow(brightness=0.7)
        loading = Pulse(color=Colors.BLUE, brightness=0.7)  # Use Pulse for loading
        pulse_waiting = Pulse(color=Colors.BLUE, brightness=0.7)
        pulse_speaking = Pulse(color=Colors.WHITE, brightness=0.7)
        fixed = Fixed(color=Colors.BLACK, brightness=0.5)
        anim_manager = AnimationManager(fixed)  # Initialize with 'fixed'
        LED_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"LED libraries not available: {e}.  LED control disabled.")
        LED_AVAILABLE = False
    except Exception as e:
        logger.error(f"Error initializing LED sequences: {e}")
        LED_AVAILABLE = False

def set_led_animation(animation_name, transition_duration=0.2):
    """Sets the LED animation if LEDs are available."""
    if not LED_AVAILABLE:
        return

    try:
        if animation_name == "rainbow":
            anim_manager.set_animation(rainbow, transition_duration)
        elif animation_name == "loading":
            anim_manager.set_animation(loading, transition_duration)
        elif animation_name == "pulse_waiting":
            anim_manager.set_animation(pulse_waiting, transition_duration)
        elif animation_name == "pulse_speaking":
            anim_manager.set_animation(pulse_speaking, transition_duration)
        elif animation_name == "fixed":
            anim_manager.set_animation(fixed, transition_duration)
        else:
            logger.warning(f"Unknown animation name: {animation_name}")
    except Exception as e:
        logger.error(f"Error setting LED animation: {e}")

def get_current_led_animation():
  """Retrieves current animation"""
  if not LED_AVAILABLE:
    return None

  try:
    return anim_manager.effective_animation()
  except Exception as e:
    logger.error(f"Error getting current LED animation: {e}")
    return None

def is_led_animation(animation_name):
    """Checks to see if current animation is the input animation"""
    if not LED_AVAILABLE:
        return False
    try:
        current_animation = anim_manager.effective_animation()

        if animation_name == "rainbow":
            return isinstance(current_animation, Rainbow)
        elif animation_name == "loading":
            return isinstance(current_animation, Pulse)  # Both loading and waiting are Pulse
        elif animation_name == "pulse_waiting":
            return isinstance(current_animation, Pulse) and current_animation.color == Colors.BLUE
        elif animation_name == "pulse_speaking":
            return isinstance(current_animation, Pulse) and current_animation.color == Colors.WHITE
        elif animation_name == "fixed":
            return isinstance(current_animation, Fixed)
        else:
            logger.warning(f"Unknown animation name for check: {animation_name}")
            return False
    except Exception as e:
        logger.exception(f"Error in is_led_animation: {e}")
        return False