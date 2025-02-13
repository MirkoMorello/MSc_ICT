from transformers import pipeline
import socket
import simpleaudio as sa
import numpy as np
import struct
import wave
import time
import os
import torch
import pyaudio
import threading
import base64
import json

# --- Configuration ----------------------
WAKE_WORD = "marvin"
PROB_THRESHOLD = 0.8
CHUNK_LENGTH_S = 2.0  # Audio chunk length for processing (in seconds)
DEBUG = True
# -----------------------------------------

class VoiceActivityDetector:
    def __init__(self, frame_duration_ms=20, threshold=10, smoothing_factor=0.8):
        self.frame_duration_ms = frame_duration_ms
        self.threshold = threshold
        self.smoothing_factor = smoothing_factor
        self.audio_buffer = []
        self.num_silent_frames = 0
        self.sample_rate = None
        self.smoothed_energy = 0
        self.voiced_frames_detected = False  # Flag to track if any voiced frames have been detected

    def set_sample_rate(self, sample_rate):
        self.sample_rate = sample_rate

    def process_audio_frame(self, frame):
        if self.sample_rate is None:
            raise ValueError("Sample rate not set. Call set_sample_rate before processing audio.")

        frame_int16 = (frame * 32767).astype(np.int16)
        energy = np.mean(np.abs(frame_int16))

        # Energy smoothing
        self.smoothed_energy = (
            self.smoothing_factor * self.smoothed_energy + (1 - self.smoothing_factor) * energy
        )

        if self.smoothed_energy > self.threshold:
            self.num_silent_frames = 0
            self.voiced_frames_detected = True 
        else:
            self.num_silent_frames += 1

        if DEBUG:
            print(f"Energy: {energy}, Smoothed Energy: {self.smoothed_energy:.2f}, Silent frames: {self.num_silent_frames}")

        self.audio_buffer.append(frame)

        # Speech end detection logic
        if self.voiced_frames_detected:
            # Standard speech end detection after at least one voiced frame
            is_speech_ended = self.num_silent_frames >= 10
        else:
            # Extended silence period if no voiced frames have been detected yet
            is_speech_ended = self.num_silent_frames >= 25

        if is_speech_ended:
            print("Speech ended")
            audio_to_send = self.audio_buffer
            # Capture whether any voiced frames were detected during this segment
            had_voiced = self.voiced_frames_detected
            self.audio_buffer = []
            self.num_silent_frames = 0
            self.voiced_frames_detected = False  # Reset the flag
            # Return an extra flag (had_voiced) along with the speech end signal and audio buffer.
            return False, audio_to_send, had_voiced  
        else:
            return True, [], None  # Continue recording

class Client:
    def __init__(self, server_ip, server_port, audio_params):
        self.server_ip = server_ip
        self.server_port = server_port
        self.audio_params = audio_params
        self.vad = VoiceActivityDetector(threshold=audio_params["threshold"])
        self.socket = None
        self.wake_word_detector = pipeline(
            "audio-classification",
            model="MIT/ast-finetuned-speech-commands-v2",
            device=0 if torch.cuda.is_available() else -1,
        )
        if WAKE_WORD not in self.wake_word_detector.model.config.label2id.keys():
            raise ValueError(
                f"Wake word {WAKE_WORD} not in set of valid class labels, pick a wake word in the set {self.wake_word_detector.model.config.label2id.keys()}."
            )
        self.vad.set_sample_rate(self.audio_params["rate"])
        self.p = pyaudio.PyAudio()
        self.stream = None

    def _connect_to_server(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.server_ip, self.server_port))

    def _disconnect_from_server(self):
        if self.socket:
            self.socket.close()

    def _send_audio(self, audio_data):
        audio_data_float32 = np.array(audio_data).astype(np.float32)
        packed_data = audio_data_float32.tobytes()
        self.socket.sendall(struct.pack("!I", len(packed_data)))
        self.socket.sendall(packed_data)

    def _receive_response(self, timeout=30):
        self.socket.settimeout(timeout)
        try:
            # Receive the size of the JSON response
            response_size_bytes = self.socket.recv(4)
            if not response_size_bytes:
                print("Connection closed by server.")
                return None, None
            response_size = struct.unpack("!I", response_size_bytes)[0]

            # Receive the JSON response
            response_data = b""
            bytes_received = 0
            while bytes_received < response_size:
                chunk = self.socket.recv(min(4096, response_size - bytes_received))
                if not chunk:
                    print("Connection closed by server during response reception.")
                    return None, None
                response_data += chunk
                bytes_received += len(chunk)

            # Parse the JSON response
            response_json = json.loads(response_data.decode())

            # Decode the Base64 audio data
            audio_base64 = response_json["audio"]
            if audio_base64:
                audio_data = np.frombuffer(base64.b64decode(audio_base64), dtype=np.float32)
            else:
                audio_data = None

            next_state = response_json["next_state"]

            return audio_data, next_state

        except socket.timeout:
            print("Timeout occurred while waiting for response data.")
            return None, None
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response: {e}")
            return None, None
        finally:
            self.socket.settimeout(None)

    def _play_audio(self, audio_data):
        # Normalize audio data
        audio_data_normalized = audio_data / np.max(np.abs(audio_data)) if np.max(np.abs(audio_data)) > 0 else audio_data

        # Audio data is a 1D array
        if audio_data_normalized.ndim > 1:
            print("Warning: Audio data has more than one dimension. Flattening to 1D array.")
            audio_data_normalized = audio_data_normalized.flatten()

        # Convert to int16 for playback
        audio_data_int16 = (audio_data_normalized * 32767).astype(np.int16)

        # Start playback
        play_obj = sa.play_buffer(
            audio_data_int16,
            num_channels=1,
            bytes_per_sample=2,
            sample_rate=self.audio_params["rate"],
        )
        play_obj.wait_done()

    def _save_audio_to_file(self, frames, filename="user_speech.wav"):
        if isinstance(frames, list):
            # Concatenate list of NumPy arrays into a single array
            frames = np.concatenate(frames)

        # Check if 'frames' is a NumPy array and has at least one dimension
        if isinstance(frames, np.ndarray) and frames.ndim > 0:
            frames_int16 = (frames * 32768).astype(np.int16)
        else:
            print(f"Error: 'frames' is not a valid NumPy array for saving. Type: {type(frames)}, Shape: {getattr(frames, 'shape', 'N/A')}")
            return

        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.audio_params["rate"])
            wf.writeframes(frames_int16.tobytes())

    def _close_with_timeout(self, stream, timeout=2):
        def close_stream():
            try:
                if stream:
                    stream.stop_stream()
                    stream.close()
            except Exception as e:
                print(f"Error closing stream: {e}")

        thread = threading.Thread(target=close_stream)
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            print("Timeout occurred while closing stream.")

    def _collect_audio_after_wake_word(self):
        print("Collecting audio after wake word...")

        while True:  # Main loop for collecting audio and handling responses
            self.vad.audio_buffer = []
            stream = self.p.open(
                format=self.audio_params["format"],
                channels=self.audio_params["channels"],
                rate=self.audio_params["rate"],
                input=True,
                frames_per_buffer=self.audio_params["chunk_size"] * 2,
            )

            speech_ended = False
            while not speech_ended:
                data = stream.read(self.audio_params["chunk_size"])
                audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Unpack the extra flag from the VAD output
                speech_continue, audio_frames, had_voiced = self.vad.process_audio_frame(audio_chunk)

                if audio_frames:
                    # If only silence was detected, skip sending audio and return to wake word detection
                    if not had_voiced:
                        print("Only silence detected. Returning to wake word detection.")
                        self._close_with_timeout(stream)
                        return

                    if DEBUG:
                        self._save_audio_to_file(audio_frames, filename="wake_word_audio.wav")

                    audio_to_send = audio_frames
                    if audio_to_send:
                        audio_data_np = np.concatenate(audio_to_send, axis=0)
                        self._send_audio(audio_data_np)
                        print("Audio sent to server. Waiting for response...")

                        response_audio, next_state = self._receive_response()
                        if response_audio is not None:
                            print(f"response_audio: {response_audio}")
                            print(f"response_audio.shape: {response_audio.shape}")
                            print(f"response_audio.dtype: {response_audio.dtype}")
                            if DEBUG:
                                self._save_audio_to_file(
                                    response_audio,
                                    filename="response_audio.wav",
                                )
                            self._play_audio(response_audio)
                            print(f"Response received and played. Next state: {next_state}")

                            self._close_with_timeout(stream)

                            if next_state == "WAKEWORD":
                                return  # Go back to wake word detection
                            elif next_state == "VAD":
                                speech_ended = True  # Go to the beginning of the main loop
                                break
                            else:
                                print("Error: Unknown next state received.")
                                return
                        else:
                            self._close_with_timeout(stream)
                            return

    def run(self):
        self._connect_to_server()
        print("Client started. Listening for wake word...")
        try:
            while True:
                stream = None
                try:
                    stream = self.p.open(
                        format=self.audio_params["format"],
                        channels=self.audio_params["channels"],
                        rate=self.audio_params["rate"],
                        input=True,
                        frames_per_buffer=self.audio_params["chunk_size"],
                    )
                    print("Listening for wake word...")

                    chunk_duration = self.audio_params["chunk_size"] / self.audio_params["rate"]
                    num_chunks_per_buffer = int(CHUNK_LENGTH_S / chunk_duration)
                    buffer = []

                    while True:
                        data = stream.read(self.audio_params["chunk_size"])
                        audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                        buffer.append(audio_chunk)

                        if len(buffer) >= num_chunks_per_buffer:
                            audio_to_process = np.concatenate(buffer)
                            predictions = self.wake_word_detector(audio_to_process)
                            prediction = predictions[0]

                            if DEBUG:
                                print(f"Prediction: {prediction}")

                            if (
                                prediction["label"] == WAKE_WORD
                                and prediction["score"] > PROB_THRESHOLD
                            ):
                                print("Wake word detected! Starting VAD.")
                                self.vad.audio_buffer = []
                                self._close_with_timeout(stream)
                                self._collect_audio_after_wake_word()
                                break  # Exit the inner while loop after processing audio

                            buffer = buffer[len(buffer) - num_chunks_per_buffer + 1:]

                except Exception as e:
                    print(f"Error during audio processing: {e}")
                    time.sleep(1)
                finally:
                    if stream:
                        self._close_with_timeout(stream)

        except KeyboardInterrupt:
            print("Stopping client...")
        finally:
            self.p.terminate()
            self._disconnect_from_server()

if __name__ == "__main__":
    audio_params = {
        "format": pyaudio.paInt16,
        "channels": 1,
        "rate": 24000,  
        "chunk_size": int(16000 * 0.5),
        "threshold": 100,
    }
    client = Client(
        server_ip="127.0.0.1", server_port=8080, audio_params=audio_params
    )
    client.run()
