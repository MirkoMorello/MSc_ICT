# This model should handle the wake word detection is we choose to train from scratch.

class SimpleWakeWordDetector:
    def __init__(self, wake_word="okay computer", threshold=0.7):
        self.wake_word = wake_word.lower()
        self.threshold = threshold

    def process_audio(self, audio_data):
        text = audio_data.lower()
        print(f"Heard: {text}")  # Debugging print
        return self.wake_word in text