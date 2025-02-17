import socket
import simpleaudio as sa
import numpy as np
import struct
import os
import wave
import torch
import pyaudio
import threading
import base64
import json
import queue
from utils import get_model, audio_amplifier
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

SERVER_ADDRESS = os.getenv("SERVER_ADDRESS", "localhost")
SERVER_PORT = int(os.getenv("SERVER_PORT", 8080))
DEPLOYMENT_MODE = os.getenv("DEPLOYMENT_MODE", "dev")

if DEPLOYMENT_MODE == "prod":
    from led_sequences.fixed import Fixed
    from led_sequences.rainbow import Rainbow
    from led_sequences.colors import Colors
    from led_sequences.animation_manager import AnimationManager
    from led_sequences.fixed import Fixed
    from led_sequences.loading import Loading
    from led_sequences.pulse import Pulse
    from led_sequences.colors import Colors
    rainbow = Rainbow(brightness=0.7)
    loading = Loading(color=Colors.BLUE, brightness=0.7, duration=7)
    pulse_waiting = Pulse(color=Colors.BLUE, brightness=0.7)
    pulse_speaking = Pulse(color=Colors.WHITE, brightness=0.7)
    fixed = Fixed(color=Colors.BLACK, brightness=0.5)
    anim_manager = AnimationManager(fixed)
    LED_AVAILABLE = True
else:
    LED_AVAILABLE = False


# --- Configuration ----------------------
WAKE_WORD = "marvin"
PROBABILITY_THRESHOLD = 0.99  # Confidence threshold
DEBUG = True
# -----------------------------------------

class VoiceActivityDetector:
    def __init__(self, frame_duration_ms=20, threshold=60, smoothing_factor=0.8):
        self.frame_duration_ms = frame_duration_ms
        self.threshold = threshold
        self.smoothing_factor = smoothing_factor
        self.audio_buffer = []
        self.num_silent_frames = 0
        self.sample_rate = None
        self.smoothed_energy = 0
        self.voiced_frames_detected = False

    def set_sample_rate(self, sample_rate):
        self.sample_rate = sample_rate

    def process_audio_frame(self, frame):
        if self.sample_rate is None:
            raise ValueError("Sample rate not set. Call set_sample_rate().")

        energy = np.mean(np.abs(frame))

        self.smoothed_energy = (
            self.smoothing_factor * self.smoothed_energy
            + (1 - self.smoothing_factor) * energy
        )

        if self.smoothed_energy > self.threshold:
            self.num_silent_frames = 0
            self.voiced_frames_detected = True
            if LED_AVAILABLE:
                if not isinstance(anim_manager.effective_animation(), Rainbow):
                    print("Changing animation to rainbow")
                    anim_manager.set_animation(rainbow, transition_duration=0.2)


        else:
            self.num_silent_frames += 1
            if LED_AVAILABLE:
                if not isinstance(anim_manager.effective_animation(), Loading):
                    print("Changing animation to Loading")
                    loading = Loading(color=Colors.BLUE, brightness=0.7, duration=3.5)
                    anim_manager.set_animation(loading, transition_duration=0.2)

        if DEBUG:
            print(
                f"Energy: {energy}, Smoothed Energy: {self.smoothed_energy:.2f}, Silent: {self.num_silent_frames}"
            )

        self.audio_buffer.append(frame)

        if self.voiced_frames_detected:
            is_speech_ended = self.num_silent_frames >= 7
        else:
            is_speech_ended = self.num_silent_frames >= 18

        if is_speech_ended:
            print("Speech ended")
            audio_to_send = self.audio_buffer
            had_voiced = self.voiced_frames_detected
            self.audio_buffer = []
            self.num_silent_frames = 0
            self.voiced_frames_detected = False
            return False, audio_to_send, had_voiced
        else:
            return True, [], None

class Client:
    def __init__(self, server_ip, server_port, audio_params, model, labels):
        self.server_ip = server_ip
        self.server_port = server_port
        self.audio_params = audio_params
        self.vad = VoiceActivityDetector(threshold=audio_params["threshold"])
        self.socket = None
        self.model = model
        self.model.eval()
        self.vad.set_sample_rate(self.audio_params["rate"])
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.wake_word_detected = False
        self.labels = labels
        self.overlap_buffer = torch.tensor([], dtype=torch.float32)
        self.overlap_samples = int(self.audio_params["rate"] * 0.25)
        self.previous_chunk = None  # Store the previous chunk
        self.led_wake_word = Fixed(Colors.BLUE) 
        self.led_waiting_wake_word = Rainbow()

    def _connect_to_server(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10)
            self.socket.connect((self.server_ip, self.server_port))
            self.socket.settimeout(None)
            print("Connected to server.")
        except socket.error as e:
            print(f"Cannot connect: {e}")
            self.socket = None
        except Exception as e:
             print(f"Unexpected error: {e}")
             self.socket = None

    def _disconnect_from_server(self):
        if self.socket:
            self.socket.close()
            self.socket = None

    def _send_audio(self, audio_data):
        if self.socket is None:
            print("Not connected to server. Cannot send audio.")
            return
        try:
            audio_data_float32 = np.concatenate(audio_data).astype(np.float32) / 32768.0 # Normalize
            packed_data = audio_data_float32.tobytes()
            self.socket.sendall(struct.pack("!I", len(packed_data)))
            self.socket.sendall(packed_data)
        except socket.error as e:
            print(f"Socket error during send: {e}")
            self._disconnect_from_server()
        except Exception as e:
            print(f"Unexpected error during send: {e}")
            self._disconnect_from_server()

    def _receive_response(self, timeout=30):
        if self.socket is None:
            print("Not connected to server. Cannot receive.")
            return None, None
        self.socket.settimeout(timeout)
        try:
            response_size_bytes = self.socket.recv(4)
            if not response_size_bytes:
                print("Connection closed.")
                return None, None

            response_size = struct.unpack("!I", response_size_bytes)[0]
            response_data = b""
            bytes_received = 0
            while bytes_received < response_size:
                chunk = self.socket.recv(min(4096, response_size - bytes_received))
                if not chunk:
                    print("Connection closed during response.")
                    return None, None
                response_data += chunk
                bytes_received += len(chunk)

            response_json = json.loads(response_data.decode())
            audio_base64 = response_json["audio"]
            if audio_base64:
                audio_data = np.frombuffer(base64.b64decode(audio_base64), dtype=np.float32)
            else:
                audio_data = None
            next_state = response_json["next_state"]
            return audio_data, next_state

        except socket.timeout:
            print("Timeout.")
            return None, None
        except (json.JSONDecodeError, struct.error) as e:
            print(f"JSON or struct error: {e}")
            return None, None
        except socket.error as e:
            print(f"Socket error: {e}")
            self._disconnect_from_server()
            return None, None
        finally:
            self.socket.settimeout(None)

    def _play_audio(self, audio_data):
        audio_data_normalized = (
            audio_data / np.max(np.abs(audio_data))
            if np.max(np.abs(audio_data)) > 0
            else audio_data
        )
        if audio_data_normalized.ndim > 1:
            print("Warning: >1 dimension. Flattening.")
            audio_data_normalized = audio_data_normalized.flatten()
        audio_data_int16 = (audio_data_normalized * 32767).astype(np.int16) 
        play_obj = sa.play_buffer(audio_data_int16, 1, 2, 24000)
        play_obj.wait_done()

    def _save_audio_to_file(self, frames, filename="user_speech.wav"):
        if isinstance(frames, list):
            frames = np.concatenate(frames)
        if isinstance(frames, np.ndarray) and frames.ndim > 0:
            frames_int16 = frames.astype(np.int16)
        else:
            print(f"Error saving: invalid array. {type(frames)}, {getattr(frames, 'shape', 'N/A')}")
            return
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.audio_params["rate"])
            wf.writeframes(frames_int16.tobytes())

    def _close_stream(self, stream):
        """Closes the PyAudio stream safely."""
        try:
            if stream and stream.is_active():
                stream.stop_stream()
                stream.close()
        except Exception as e:
            print(f"Error closing stream: {e}")

    def _collect_audio_after_wake_word(self):
        print("Collecting audio after wake word...")
        while True:
            self.vad.audio_buffer = []
            stream = self.p.open(
                format=pyaudio.paInt16,
                channels=self.audio_params["channels"],
                rate=self.audio_params["rate"],
                input=True,
                frames_per_buffer=self.audio_params["chunk_size"] * 2,
            )
            try:
                speech_ended = False
                while not speech_ended:
                    data = stream.read(self.audio_params["chunk_size"], exception_on_overflow=False)
                    audio_chunk = np.frombuffer(data, dtype=np.int16)
                    if DEPLOYMENT_MODE == 'prod':
                        audio_chunk = audio_amplifier(audio_chunk)
                    speech_continue, audio_frames, had_voiced = self.vad.process_audio_frame(audio_chunk)

                    if audio_frames:
                        if not had_voiced:
                            print("Only silence. Back to wake word.")
                            self._close_stream(stream)
                            return
                        if DEBUG:
                             self._save_audio_to_file(audio_frames, "post_wake.wav")
                        self._send_audio(audio_frames)
                        print("Audio sent. Waiting...")
                        if LED_AVAILABLE:
                            print("changing animation to pulse\n")
                            anim_manager.set_animation(pulse_waiting, transition_duration=0.2)

                        response_audio, next_state = self._receive_response()
                        if response_audio is not None:
                            if DEBUG:
                                self._save_audio_to_file(response_audio, "response.wav")
                            if LED_AVAILABLE:
                                anim_manager.set_animation(pulse_speaking, transition_duration=0.2)
                            self._play_audio(response_audio)
                            print(f"Response played. Next: {next_state}")
                            if next_state == "WAKEWORD":
                                self._close_stream(stream)
                                return
                            elif next_state == "VAD":
                                continue
                            else:
                                print("Error: Unknown state.")
                                self._close_stream(stream)
                                return
                        else: # no response
                            self._close_stream(stream)
                            return
            finally:
                self._close_stream(stream)

    def _process_audio_chunk(self, audio_chunk):
        """Processes a chunk, returns (predicted_index, probability)."""
        audio_length = torch.tensor([audio_chunk.size(1)])
        with torch.no_grad():
            logits = self.model(audio_chunk, audio_length)
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class_index = torch.argmax(probabilities, dim=-1).item()
            probability = probabilities[0, predicted_class_index].item()
        return predicted_class_index, probability

    def _wake_word_detection_loop(self):
        """Records audio, checks for wake word (using current and previous chunks)."""
        stream = self.p.open(
            format=pyaudio.paInt16,
            channels=self.audio_params["channels"],
            rate=self.audio_params["rate"],
            input=True,
            frames_per_buffer=self.audio_params["chunk_size"],
        )
        print("Listening for wake word...")
        print(f"Initial overlap_buffer size: {self.overlap_buffer.size()}")

        # --- FLUSH INITIAL AUDIO DATA ---
        print("Flushing initial audio data...")
        initial_flush_duration = 0.1  # Seconds
        initial_flush_samples = int(self.audio_params["rate"] * initial_flush_duration)
        stream.read(initial_flush_samples, exception_on_overflow=False)
        print("Flush complete.")
        # ----------------------------------

        try:
            while not self.wake_word_detected:
                if LED_AVAILABLE:
                    if not isinstance(anim_manager.next_animation, Fixed):
                        if not isinstance(anim_manager.current_animation, Fixed):
                            print("Changing animation to rainbow")
                            anim_manager.set_animation(fixed, transition_duration=0.2)

                data = stream.read(self.audio_params["chunk_size"], exception_on_overflow=False)
                audio_array = np.frombuffer(data, dtype=np.int16)
                if DEPLOYMENT_MODE == 'prod':
                    audio_array = audio_amplifier(audio_chunk = audio_array, factor = 15)
                current_chunk = (torch.from_numpy(audio_array).unsqueeze(0).to(torch.float32) / 32768.0)

                # --- Combine with Previous Chunk (if available) ---
                if self.previous_chunk is not None:
                    combined_input = torch.cat((self.previous_chunk, current_chunk), dim=1)
                else:
                    combined_input = current_chunk  # First chunk, use only current

				# --- Process combined chunk ---
                if combined_input.size(1) >= self.audio_params["chunk_size"]:
                    predicted_index, probability = self._process_audio_chunk(combined_input)
                    predicted_label = self.labels[predicted_index]

                    if DEBUG:
                        print(f"Predicted label: {predicted_label}, Probability: {probability:.4f}")

                    if predicted_label.lower() == WAKE_WORD and probability >= PROBABILITY_THRESHOLD:
                        print("Wake word detected!")
                        self.wake_word_detected = True
                        break
                # --- Update previous_chunk ---
                self.previous_chunk = current_chunk

                # --- overlap buffer is not used in this logic ---

        finally:
            self._close_stream(stream)
            self.previous_chunk = None  # Reset for next loop

    def run(self):
        """Main client loop."""
        self._connect_to_server()
        if self.socket is None:
             return
        print("Client started. Listening...")
        try:
            while True:
                self.wake_word_detected = False
                self.overlap_buffer = torch.tensor([], dtype=torch.float32) #reset
                self.previous_chunk = None  # Reset previous_chunk
                self._wake_word_detection_loop()
                if self.wake_word_detected:
                    self._collect_audio_after_wake_word()
        except KeyboardInterrupt:
            print("Stopping.")
        finally:
            self.p.terminate()
            self._disconnect_from_server()

if __name__ == "__main__":
    audio_params = {
        "format": pyaudio.paInt16,
        "channels": 1,
        "rate": 16000,
        "chunk_size": int(16000 * 0.5),  # 0.5 seconds
        "threshold": 900,
    }

    model = get_model(path = "../best_model.pth")
    if model is None:
        print("Error: Model loading failed.")
        exit()

    labels = [
        'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
        'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin',
        'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six',
        'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'
    ]

    client = Client(
        server_ip=SERVER_ADDRESS,  #  your server's IP
        server_port=SERVER_PORT,
        audio_params=audio_params,
        model=model,
        labels=labels,
    )
    client.run()
