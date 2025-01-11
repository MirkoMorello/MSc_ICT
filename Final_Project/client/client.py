import pyaudio
import socket
import time
import numpy as np
import struct
import wave
# Placeholder for a more advanced wake word detection algorithm later
class SimpleWakeWordDetector:
    def __init__(self, wake_word="okay computer", threshold=0.7):
        self.wake_word = wake_word.lower()
        self.threshold = threshold

    def process_audio(self, audio_data):
        # This is a placeholder. Replace with a real wake word model later.
        text = audio_data.lower()
        print(f"Heard: {text}")  # Debugging print
        return self.wake_word in text

class VoiceActivityDetector:
    def __init__(self, frame_duration_ms=20, padding_duration_ms=300, threshold=0.6):
        self.frame_duration_ms = frame_duration_ms
        self.padding_duration_ms = padding_duration_ms
        self.threshold = threshold
        self.padding_frames = int(padding_duration_ms / frame_duration_ms)
        self.audio_buffer = []
        self.num_voiced_frames = 0
        self.num_silent_frames = 0
        self.in_speech = False

    def process_audio_frame(self, frame):
        # Simple energy-based detection
        energy = np.mean(np.abs(frame))

        if energy > self.threshold:
            self.num_voiced_frames += 1
            self.num_silent_frames = 0
        else:
            self.num_silent_frames += 1
            self.num_voiced_frames = 0

        # Add frame to buffer
        self.audio_buffer.append(frame)

        # Check for speech start
        if not self.in_speech and self.num_voiced_frames > 5:
            self.in_speech = True
            print("Speech started")
            return True, self.audio_buffer

        # Check for speech end
        elif self.in_speech and self.num_silent_frames > self.padding_frames:
            self.in_speech = False
            print("Speech ended")
            # Remove padding from the end
            audio_to_send = self.audio_buffer[:-self.padding_frames]
            self.audio_buffer = []
            self.num_voiced_frames = 0
            return False, audio_to_send

        return None, []

class Client:
    def __init__(self, server_ip, server_port, audio_params):
        self.server_ip = server_ip
        self.server_port = server_port
        self.audio_params = audio_params
        self.wake_word_detector = SimpleWakeWordDetector()
        self.vad = VoiceActivityDetector()
        self.stream = None
        self.socket = None
        self.p = pyaudio.PyAudio()

    def _open_audio_stream(self):
        self.stream = self.p.open(
            format=self.audio_params['format'],
            channels=self.audio_params['channels'],
            rate=self.audio_params['rate'],
            input=True,
            frames_per_buffer=self.audio_params['chunk_size']
        )

    def _close_audio_stream(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

    def _connect_to_server(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.server_ip, self.server_port))

    def _disconnect_from_server(self):
        if self.socket:
            self.socket.close()

    def _send_audio(self, audio_data):
        # Pack the audio data and send it
        packed_data = struct.pack(f"!{len(audio_data)}h", *audio_data)
        self.socket.sendall(packed_data)

    def _receive_audio(self):
        # Placeholder for receiving audio from the server
        # You'll need to define a protocol for sending audio back
        # For example, you could send the size of the audio data first,
        # followed by the actual data.
        pass
    
    def _save_audio_to_file(self, frames, filename="user_speech.wav"):
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.audio_params['channels'])
            wf.setsampwidth(self.p.get_sample_size(self.audio_params['format']))
            wf.setframerate(self.audio_params['rate'])
            wf.writeframes(b''.join(frames))

    def run(self):
        self._connect_to_server()
        self._open_audio_stream()

        print("Client started. Listening for wake word...")
        try:
            while True:
                audio_chunk = self.stream.read(self.audio_params['chunk_size'], exception_on_overflow=False)
                # Convert audio chunk to numpy array of int16
                audio_data_np = np.frombuffer(audio_chunk, dtype=np.int16)
                # Convert to mono if stereo
                if self.audio_params['channels'] == 2:
                    audio_data_np = audio_data_np.reshape(-1, 2).mean(axis=1).astype(np.int16)

                # Concatenate audio data to a list of frames
                frames = [audio_data_np]

                # Placeholder for wake word detection
                if self.wake_word_detector.process_audio(audio_chunk.decode('iso-8859-1', errors='ignore')):
                    print("Wake word detected! Starting VAD.")
                    self.vad.in_speech = True  # Reset VAD
                    self.vad.audio_buffer = []
                    while self.vad.in_speech:
                        audio_chunk = self.stream.read(self.audio_params['chunk_size'], exception_on_overflow=False)
                        audio_data_np = np.frombuffer(audio_chunk, dtype=np.int16)
                        # Convert to mono if stereo
                        if self.audio_params['channels'] == 2:
                            audio_data_np = audio_data_np.reshape(-1, 2).mean(axis=1).astype(np.int16)
                        
                        speech_started, audio_frames = self.vad.process_audio_frame(audio_data_np)
                        frames.extend(audio_frames)  # Store all frames until speech ends
                        
                        if speech_started == False:
                            self._save_audio_to_file(frames)
                            self._send_audio([val for frame in audio_frames for val in frame])  # Flatten list of frames
                            frames = []  # Reset frames
                            self.vad.in_speech = False
                            print("Audio sent to server. Waiting for response...")
        except KeyboardInterrupt:
            print("Stopping client...")
        finally:
            self._close_audio_stream()
            self._disconnect_from_server()
            self.p.terminate()

if __name__ == "__main__":
    audio_params = {
        'format': pyaudio.paInt16,
        'channels': 1,  # Mono
        'rate': 16000,
        'chunk_size': 1024
    }
    print('')
    client = Client(server_ip="127.0.0.1", server_port=12345, audio_params=audio_params)
    client.run() 