from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_microphone_live
import socket
import simpleaudio as sa
import numpy as np
import struct
import wave
import time
import os
import pyaudio
import threading

# --- Configuration ----------------------
WAKE_WORD = "marvin"
PROB_THRESHOLD = 0.8  # threshold for wake word detection
CHUNK_LENGTH_S = .5  # audio chunk length for the classifier (in seconds)
STREAM_CHUNK_S = .5  # 
DEBUG = True
# -----------------------------------------

class VoiceActivityDetector:
    def __init__(self, frame_duration_ms=20, padding_duration_ms=300, threshold=400):
        self.frame_duration_ms = frame_duration_ms
        self.padding_duration_ms = padding_duration_ms
        self.threshold = threshold
        self.padding_frames = int(
            padding_duration_ms / frame_duration_ms
        )  # Example: 300 / 20 = 15
        self.audio_buffer = []
        self.num_voiced_frames = 0
        self.num_silent_frames = 0
        self.in_speech = False
        self.sample_rate = None

    def set_sample_rate(self, sample_rate):
        self.sample_rate = sample_rate

    def process_audio_frame(self, frame):
        if self.sample_rate is None:
            raise ValueError(
                "Sample rate not set. Call set_sample_rate before processing audio."
            )

        # Convert frame to int16 for energy calculation
        frame_int16 = (frame * 32768).astype(np.int16)

        # Simple energy-based detection
        energy = np.mean(np.abs(frame_int16))

        if energy > self.threshold:
            self.num_voiced_frames += 1
            self.num_silent_frames = 0
        else:
            self.num_silent_frames += 1
            self.num_voiced_frames = 0

        if DEBUG:
            print(
                f"Energy: {energy}, Voiced frames: {self.num_voiced_frames}, Silent frames: {self.num_silent_frames}"
            )

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
            audio_to_send = self.audio_buffer[: -self.padding_frames]
            self.audio_buffer = []
            self.num_voiced_frames = 0
            return False, audio_to_send

        return None, []

class Client:
    def __init__(self, server_ip, server_port, audio_params):
        self.server_ip = server_ip
        self.server_port = server_port
        self.audio_params = audio_params
        self.vad = VoiceActivityDetector()
        self.socket = None
        # Initialize the wake word detection pipeline
        self.wake_word_detector = pipeline(
            "audio-classification",
            model="MIT/ast-finetuned-speech-commands-v2",
            device=0,  # if you have a GPU, otherwise remove device
        )
        if WAKE_WORD not in self.wake_word_detector.model.config.label2id.keys():
            raise ValueError(
                f"Wake word {WAKE_WORD} not in set of valid class labels, pick a wake word in the set {self.wake_word_detector.model.config.label2id.keys()}."
            )
        self.vad.set_sample_rate(
            self.wake_word_detector.feature_extractor.sampling_rate
        )
        self.speech_start_time = None
        self.mic_stream = None # Add a stream attribute

    def _connect_to_server(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.server_ip, self.server_port))

    def _disconnect_from_server(self):
        if self.socket:
            self.socket.close()

    def _send_audio(self, audio_data):
        # Convert the audio data to float32 before packing
        audio_data_float32 = np.array(audio_data).astype(np.float32)
        # Flatten the audio data array and pack it
        packed_data = audio_data_float32.tobytes()
        # Send the size of the data first
        self.socket.sendall(struct.pack("!I", len(packed_data)))
        # Then send the data
        self.socket.sendall(packed_data)

    def _receive_audio(self, timeout=10):
        # Set a timeout for receiving data
        self.socket.settimeout(timeout)
        try:
            # Receive the size of the audio data first (an integer)
            data_size_bytes = self.socket.recv(4)
            if not data_size_bytes:
                print("Connection closed by server.")
                return None
            data_size = struct.unpack("!I", data_size_bytes)[0]

            # Receive the actual audio data
            audio_data = b""
            bytes_received = 0
            while bytes_received < data_size:
                chunk = self.socket.recv(
                    min(4096, data_size - bytes_received)
                )  # Receive in chunks
                if not chunk:
                    print("Connection closed by server during audio reception.")
                    return None
                audio_data += chunk
                bytes_received += len(chunk)

            # Convert the received bytes to a NumPy array of float32
            audio_np = np.frombuffer(audio_data, dtype=np.float32)
            return audio_np
        except socket.timeout:
            print("Timeout occurred while waiting for audio data.")
            return None
        finally:
            self.socket.settimeout(None)  # Reset the timeout

    def _play_audio(self, audio_data):
        # Ensure the audio data is in the correct format (int16)
        audio_data_int16 = (audio_data * 32767).astype(np.int16)

        # Start playback
        play_obj = sa.play_buffer(
            audio_data_int16,
            num_channels=1,
            bytes_per_sample=2,
            sample_rate=16000,
        )

        # Wait for playback to finish
        play_obj.wait_done()

    def _save_audio_to_file(self, frames, filename="user_speech.wav"):
        # Convert frames to numpy array of int16 for saving
        frames_int16 = (np.concatenate(frames) * 32768).astype(np.int16)

        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)  # Assuming mono audio
            wf.setsampwidth(2)  # 2 bytes for int16
            wf.setframerate(self.wake_word_detector.feature_extractor.sampling_rate)
            wf.writeframes(frames_int16.tobytes())
            
    def _stop_mic_stream(self):
        if self.mic_stream:
            print("Closing microphone stream...")
            try:
                self.mic_stream.close()  # Use the close method of the iterator
            except Exception as e:
                print(f"Error closing microphone stream: {e}")
            self.mic_stream = None

    def _close_with_timeout(self, mic, timeout=2):
        def close_mic():
            try:
                mic.close()
            except Exception as e:
                print(f"Error closing microphone stream: {e}")

        thread = threading.Thread(target=close_mic)
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            print("Timeout occurred while closing microphone stream.")

    def _collect_audio_after_wake_word(self, sampling_rate):
            print("Collecting audio after wake word...")
            frames = []
            self.vad.in_speech = True
            self.vad.num_voiced_frames = 0
            self.vad.num_silent_frames = 0
            self.vad.audio_buffer = []

            speech_mic = ffmpeg_microphone_live(
                sampling_rate=sampling_rate,
                chunk_length_s=CHUNK_LENGTH_S,
                stream_chunk_s=STREAM_CHUNK_S,
            )

            for j, audio_chunk in enumerate(speech_mic):
                if DEBUG:
                    print(f"VAD - {j}: Processing chunk")
                speech_started, audio_frames = self.vad.process_audio_frame(audio_chunk["raw"])
                print(f"VAD - {j}: speech_started={speech_started}, audio_frames={audio_frames}, type(audio_frames)={type(audio_frames)}")

                # Only extend frames if audio_frames is not empty and contains arrays
                if audio_frames and all(isinstance(f, np.ndarray) for f in audio_frames):
                    print(f"   - Inside if audio_frames: {audio_frames}")
                    frames.extend(audio_frames)
                    print(f"   - frames after extend: {frames}")

                if speech_started == False:
                    print(f"Speech ended: frames={frames}")
                    if DEBUG:
                        self._save_audio_to_file(frames, filename="wake_word_audio.wav")

                    audio_to_send = frames

                    if audio_to_send:
                        print(f"About to concatenate: audio_to_send={audio_to_send}")
                        # Safeguard against zero-dimensional arrays
                        if all(f.ndim > 0 for f in audio_to_send):
                            audio_data_np = np.concatenate(audio_to_send, axis=0)
                            self._send_audio(audio_data_np)
                            print("Audio sent to server. Waiting for response...")

                            response_audio = self._receive_audio()
                            if response_audio is not None:
                                if DEBUG:
                                    self._save_audio_to_file(
                                        response_audio,
                                        filename="response_audio.wav",
                                    )
                                print("Response received.")
                        else:
                            print("Cannot concatenate: audio_to_send contains zero-dimensional arrays")
                    else:
                        print("No audio to send.")

                    # Clear frames and other variables
                    frames = []  # Clear frames
                    self.vad.audio_buffer = []
                    self.vad.in_speech = False
                    self.speech_start_time = None
                    self._close_with_timeout(speech_mic)
                    return

    def run(self):
        self._connect_to_server()
        print("Client started. Listening for wake word...")
        try:
            sampling_rate = self.wake_word_detector.feature_extractor.sampling_rate

            while True:
                mic = None
                try:
                    mic = ffmpeg_microphone_live(
                        sampling_rate=sampling_rate,
                        chunk_length_s=CHUNK_LENGTH_S,
                        stream_chunk_s=STREAM_CHUNK_S,
                    )
                    print("Listening for wake word...")

                    for i, chunk in enumerate(mic):
                        predictions = self.wake_word_detector(chunk)
                        prediction = predictions[0]

                        if DEBUG:
                            print(f"{i}: {prediction}")

                        if (
                            prediction["label"] == WAKE_WORD
                            and prediction["score"] > PROB_THRESHOLD
                        ):
                            print("Wake word detected! Starting VAD.")
                            self.speech_start_time = time.time()
                            self.vad.in_speech = True
                            self.vad.audio_buffer = []
                            frames = []

                            self._close_with_timeout(mic)  # Close with timeout
                            self._collect_audio_after_wake_word(sampling_rate)
                            break

                except Exception as e:
                    print(f"Error during audio processing: {e}")
                    time.sleep(1)
                finally:
                    if mic:
                        self._close_with_timeout(mic)

        except KeyboardInterrupt:
            print("Stopping client...")
        finally:
            self._disconnect_from_server()

if __name__ == "__main__":
    audio_params = {
        "format": pyaudio.paInt16,
        "channels": 1,  # Mono
        "rate": 16000,  # 16 kHz
        "chunk_size": 1024,
    }
    client = Client(
        server_ip="127.0.0.1", server_port=8080, audio_params=audio_params
    )
    client.run()