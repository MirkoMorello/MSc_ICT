from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_microphone_live
import pyaudio
import socket
import simpleaudio as sa
import time
import numpy as np
import struct
import wave
import torch


# --- Configuration ----------------------
WAKE_WORD = "marvin"  
PROB_THRESHOLD = 0.8  # threshold for wake word detection
CHUNK_LENGTH_S = 2.0   # audio chunk length for the classifier (in seconds)
STREAM_CHUNK_S = 0.25  # chunk size for streaming to the classifier
DEBUG = True          
# -----------------------------------------



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
        self.sample_rate = None

    def set_sample_rate(self, sample_rate):
        self.sample_rate = sample_rate

    def process_audio_frame(self, frame):
        if self.sample_rate is None:
            raise ValueError("Sample rate not set. Call set_sample_rate before processing audio.")

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
        self.vad = VoiceActivityDetector()
        self.socket = None
        # Initialize the wake word detection pipeline
        self.wake_word_detector = pipeline(
            "audio-classification", model="MIT/ast-finetuned-speech-commands-v2", device=0  # if you have a GPU, otherwise remove device
        )
        if WAKE_WORD not in self.wake_word_detector.model.config.label2id.keys():
            raise ValueError(
                f"Wake word {WAKE_WORD} not in set of valid class labels, pick a wake word in the set {self.wake_word_detector.model.config.label2id.keys()}."
            )
        self.vad.set_sample_rate(self.wake_word_detector.feature_extractor.sampling_rate)

    def _connect_to_server(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.server_ip, self.server_port))

    def _disconnect_from_server(self):
        if self.socket:
            self.socket.close()

    def _send_audio(self, audio_data):
        # Convert the audio data to int16 before packing
        audio_data_int16 = (np.array(audio_data) * 32768).astype(np.int16)
        # Flatten the audio data array and pack it
        packed_data = struct.pack(f"!{len(audio_data_int16.ravel())}h", *audio_data_int16.ravel())
        self.socket.sendall(packed_data)

    def _receive_audio(self):
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
            chunk = self.socket.recv(min(4096, data_size - bytes_received))  # Receive in chunks
            if not chunk:
                print("Connection closed by server during audio reception.")
                return None
            audio_data += chunk
            bytes_received += len(chunk)

        # Convert the received bytes to a NumPy array of float32
        audio_np = np.frombuffer(audio_data, dtype=np.float32)
        
        return audio_np

    def _play_audio(self, audio_data):

        # Ensure the audio data is in the correct format (int16)
        audio_data_int16 = (audio_data * 32767).astype(np.int16)

        # Start playback
        play_obj = sa.play_buffer(audio_data_int16, num_channels=1, bytes_per_sample=2, sample_rate=16000)

        # Wait for playback to finish
        play_obj.wait_done()

    def _save_audio_to_file(self, frames, filename="user_speech.wav"):
        # Convert frames to numpy array of int16 for saving
        frames_int16 = (np.concatenate(frames) * 32768).astype(np.int16)
    
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)  # Assuming mono audio
            wf.setsampwidth(2)  # 2 bytes for int16
            wf.setframerate(self.wake_word_detector.feature_extractor.sampling_rate)
            wf.writeframes(frames_int16.tobytes())

    def run(self):
            self._connect_to_server()

            print("Client started. Listening for wake word...")
            try:
                sampling_rate = self.wake_word_detector.feature_extractor.sampling_rate
                mic = ffmpeg_microphone_live(
                    sampling_rate=sampling_rate,
                    chunk_length_s=CHUNK_LENGTH_S,
                    stream_chunk_s=STREAM_CHUNK_S,
                )
                
                print("Listening for wake word...")
                for chunk in mic:
                    # Process the chunk through the wake word detector
                    predictions = self.wake_word_detector(chunk)
                    prediction = predictions[0]
                    
                    if DEBUG:
                        print(prediction)
                    
                    # Check for wake word
                    if prediction["label"] == WAKE_WORD and prediction["score"] > PROB_THRESHOLD:
                        print("Wake word detected! Starting VAD.")
                        self.vad.in_speech = True  # Reset VAD
                        self.vad.audio_buffer = []
                        frames = []
                        
                        # Continue collecting audio in chunks until speech ends
                        for audio_chunk in mic:
                            if not self.vad.in_speech:
                                break  # Exit loop if speech has ended

                            speech_started, audio_frames = self.vad.process_audio_frame(audio_chunk["raw"])
                            frames.extend(audio_frames)  # Store all frames until speech ends

                            if speech_started == False:
                                self._save_audio_to_file(frames)
                                # Convert list of numpy arrays to a single numpy array before flattening
                                audio_data_np = np.concatenate(audio_frames, axis=0)
                                self._send_audio(audio_data_np.flatten())
                                frames = []  # Reset frames
                                print("Audio sent to server. Waiting for response...")

                                # Receive and play the audio response
                                response_audio = self._receive_audio()
                                if response_audio is not None:
                                    self._play_audio(response_audio)
                                    print("Response played.")

            except KeyboardInterrupt:
                print("Stopping client...")
            finally:
                self._disconnect_from_server()

if __name__ == "__main__":
    audio_params = {
        'format': pyaudio.paInt16,
        'channels': 1,  # Mono
        'rate': 16000,  # 16 kHz
        'chunk_size': 1024
    }
    client = Client(server_ip="127.0.0.1", server_port=12345, audio_params=audio_params)
    client.run()