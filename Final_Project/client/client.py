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
import time
import logging
from .utils.vad import VoiceActivityDetector
from .utils.audio_utils import get_model, audio_amplifier
from .config import (
    SERVER_ADDRESS, SERVER_PORT, AUDIO_FORMAT, AUDIO_CHANNELS, AUDIO_RATE,
    AUDIO_CHUNK_SIZE, VAD_THRESHOLD, WAKE_WORD, PROBABILITY_THRESHOLD,
    DEPLOYMENT_MODE, AMPLIFICATION_FACTOR_WAKE_WORD, MODEL_PATH, LABELS
)
from .utils.led_utils import set_led_animation, LED_AVAILABLE, is_led_animation, get_current_led_animation
from .utils import logging_utils  # Corrected import

logger = logging_utils.get_logger(__name__)

class Client:
    def __init__(self, server_ip, server_port, audio_params, model, labels):
        self.server_ip = server_ip
        self.server_port = server_port
        self.audio_params = audio_params
        self.vad = VoiceActivityDetector(threshold=audio_params["threshold"])
        self.socket = None
        self.model = model
        if self.model:
            self.model.eval()
        self.vad.set_sample_rate(self.audio_params["rate"])
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.wake_word_detected = False
        self.labels = labels
        self.overlap_buffer = torch.tensor([], dtype=torch.float32)
        self.overlap_samples = int(self.audio_params["rate"] * 0.25)
        self.previous_chunk = None

    def _connect_to_server(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10)
            self.socket.connect((self.server_ip, self.server_port))
            self.socket.settimeout(None)
            logger.info("Connected to server.")
        except socket.error as e:
            logger.error(f"Cannot connect: {e}")
            self.socket = None
            raise
        except Exception as e:
            logger.exception(f"Unexpected error during connection: {e}")
            self.socket = None
            raise

    def _disconnect_from_server(self):
        if self.socket:
            try:
                self.socket.close()
                logger.info("Disconnected from server.")
            except socket.error as e:
                logger.error(f"Error disconnecting: {e}")
            finally:
                self.socket = None

    def _send_audio(self, audio_data):
        if self.socket is None:
            logger.warning("Not connected to server. Cannot send audio.")
            return
        try:
            audio_data_float32 = np.concatenate(audio_data).astype(np.float32) / 32768.0
            packed_data = audio_data_float32.tobytes()
            self.socket.sendall(struct.pack("!I", len(packed_data)))
            self.socket.sendall(packed_data)
            logger.debug(f"Sent {len(packed_data)} bytes of audio data.")
        except socket.error as e:
            logger.error(f"Socket error during send: {e}")
            self._disconnect_from_server()
        except Exception as e:
            logger.exception(f"Unexpected error during send: {e}")
            self._disconnect_from_server()

    def _receive_response(self, timeout=30):
        if self.socket is None:
            logger.warning("Not connected to server. Cannot receive.")
            return None, None
        self.socket.settimeout(timeout)
        try:
            response_size_bytes = self.socket.recv(4)
            if not response_size_bytes:
                logger.info("Connection closed by server.")
                return None, None

            response_size = struct.unpack("!I", response_size_bytes)[0]
            response_data = b""
            bytes_received = 0
            while bytes_received < response_size:
                chunk = self.socket.recv(min(4096, response_size - bytes_received))
                if not chunk:
                    logger.warning("Connection closed during response.")
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
            logger.warning("Timeout receiving response.")
            return None, None
        except (json.JSONDecodeError, struct.error) as e:
            logger.error(f"JSON or struct error: {e}")
            return None, None
        except socket.error as e:
            logger.error(f"Socket error: {e}")
            self._disconnect_from_server()
            return None, None
        finally:
            self.socket.settimeout(None)

    def _play_audio(self, audio_data):
        try:
            audio_data_normalized = (
                audio_data / np.max(np.abs(audio_data))
                if np.max(np.abs(audio_data)) > 0
                else audio_data
            )
            if audio_data_normalized.ndim > 1:
                logger.warning(">1 dimension. Flattening.")
                audio_data_normalized = audio_data_normalized.flatten()
            audio_data_int16 = (audio_data_normalized * 32767).astype(np.int16)
            play_obj = sa.play_buffer(audio_data_int16, 1, 2, 24000)  # Use 24000 consistently
            play_obj.wait_done()
        except Exception as e:
            logger.exception(f"Error playing audio: {e}")

    def _save_audio_to_file(self, frames, filename="user_speech.wav"):
        try:
            if isinstance(frames, list):
                frames = np.concatenate(frames)
            if isinstance(frames, np.ndarray) and frames.ndim > 0:
                frames_int16 = frames.astype(np.int16)
            else:
                logger.error(f"Error saving: invalid array. {type(frames)}, {getattr(frames, 'shape', 'N/A')}")
                return

            with wave.open(filename, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.audio_params["rate"])
                wf.writeframes(frames_int16.tobytes())
            logger.debug(f"Saved audio to {filename}")
        except Exception as e:
             logger.exception(f"Error while saving audio to file")

    def _close_stream(self, stream):
        try:
            if stream and stream.is_active():
                stream.stop_stream()
                stream.close()
                logger.debug("Stream closed.")
        except Exception as e:
            logger.error(f"Error closing stream: {e}")

    def _collect_audio_after_wake_word(self):
        logger.info("Collecting audio after wake word...")
        temp_files = []  # Keep track of temp files
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
                        audio_chunk = audio_amplifier(audio_chunk, factor = AMPLIFICATION_FACTOR_VAD)
                    speech_continue, audio_frames, had_voiced = self.vad.process_audio_frame(audio_chunk)

                    if audio_frames:
                        if not had_voiced:
                            logger.info("Only silence. Back to wake word.")
                            self._close_stream(stream)
                            return  # Go back to wake word detection

                        post_wake_file = "post_wake_temp.wav"  # Use a consistent name
                        temp_files.append(post_wake_file)
                        self._save_audio_to_file(audio_frames, post_wake_file)

                        self._send_audio(audio_frames)
                        logger.info("Audio sent. Waiting for response...")

                        if LED_AVAILABLE:
                            set_led_animation("pulse_waiting", transition_duration=0.2)

                        response_audio, next_state = self._receive_response()

                        if response_audio is not None:
                            response_file = "response_temp.wav"
                            temp_files.append(response_file)
                            self._save_audio_to_file(response_audio, response_file)

                            if LED_AVAILABLE:
                                 set_led_animation("pulse_speaking", transition_duration=0.2)
                            self._play_audio(response_audio)
                            logger.info(f"Response played. Next state: {next_state}")

                            if next_state == "WAKEWORD":
                                self._close_stream(stream)
                                return  # Exit this function, go back to wake word
                            elif next_state == "VAD":
                                continue  # Continue recording in VAD mode
                            else:
                                logger.error("Error: Unknown state.")
                                self._close_stream(stream)
                                return
                        else:
                            # No response received, go back to wake-word
                            self._close_stream(stream)
                            return

            finally:
                self._close_stream(stream)
                # Clean up temporary files
                for temp_file in temp_files:
                    try:
                        os.remove(temp_file)
                        logger.debug(f"Deleted temporary file: {temp_file}")
                    except FileNotFoundError:
                        pass  # File might have already been deleted
                    except Exception as e:
                        logger.error(f"Error deleting {temp_file}: {e}")

    def _process_audio_chunk(self, audio_chunk):
        audio_length = torch.tensor([audio_chunk.size(1)])
        with torch.no_grad():
            logits = self.model(audio_chunk, audio_length)
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class_index = torch.argmax(probabilities, dim=-1).item()
            probability = probabilities[0, predicted_class_index].item()
        return predicted_class_index, probability

    def _wake_word_detection_loop(self):
        stream = self.p.open(
            format=pyaudio.paInt16,
            channels=self.audio_params["channels"],
            rate=self.audio_params["rate"],
            input=True,
            frames_per_buffer=self.audio_params["chunk_size"],
        )
        logger.info("Listening for wake word...")
        logger.debug(f"Initial overlap_buffer size: {self.overlap_buffer.size()}")

        logger.debug("Flushing initial audio data...")
        initial_flush_duration = 0.1
        initial_flush_samples = int(self.audio_params["rate"] * initial_flush_duration)
        stream.read(initial_flush_samples, exception_on_overflow=False)
        logger.debug("Flush complete.")

        try:
            while not self.wake_word_detected:
                if LED_AVAILABLE:
                    if not is_led_animation("fixed"):
                        set_led_animation("fixed", transition_duration=0.2)

                data = stream.read(self.audio_params["chunk_size"], exception_on_overflow=False)
                audio_array = audio_amplifier(np.frombuffer(data, dtype=np.int16), factor=AMPLIFICATION_FACTOR_WAKE_WORD)
                current_chunk = (torch.from_numpy(audio_array).unsqueeze(0).to(torch.float32) / 32768.0)

                if self.previous_chunk is not None:
                    combined_input = torch.cat((self.previous_chunk, current_chunk), dim=1)
                else:
                    combined_input = current_chunk

                if combined_input.size(1) >= self.audio_params["chunk_size"]:
                    predicted_index, probability = self._process_audio_chunk(combined_input)
                    predicted_label = self.labels[predicted_index]

                    if True: # Changed DEBUG for flag.
                        logger.debug(f"Predicted label: {predicted_label}, Probability: {probability:.4f}")

                    if predicted_label.lower() == WAKE_WORD and probability >= PROBABILITY_THRESHOLD:
                        logger.info("Wake word detected!")
                        self.wake_word_detected = True
                        break

                self.previous_chunk = current_chunk

        finally:
            self._close_stream(stream)
            self.previous_chunk = None

    def run(self):
        is_connected = False
        while not is_connected:
            try:
                self._connect_to_server()
                is_connected = True
            except socket.error:
                is_connected = False
                logger.info("Connection to server failed... trying again...")
                time.sleep(1)
        if self.socket is None:
             return

        with wave.open("./client/utils/server_connected_msg.wav", 'rb') as wf:
            audio_data = wf.readframes(wf.getnframes())
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            self._play_audio(audio_array)

        logger.info("Client started. Listening...")
        try:
            while True:
                self.wake_word_detected = False
                self.overlap_buffer = torch.tensor([], dtype=torch.float32)
                self.previous_chunk = None
                self._wake_word_detection_loop()
                if self.wake_word_detected:
                    self._collect_audio_after_wake_word()
        except KeyboardInterrupt:
            logger.info("Stopping.")
        finally:
            self.p.terminate()
            self._disconnect_from_server()