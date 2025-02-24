import socket
import struct
import json
import base64
import logging
import time
from typing import Optional
import numpy as np
from .audio_processing import AudioProcessor 
from .utils.llm_handler import LLMHandler  
from .utils.tts_handler import TTSHandler  
from .utils.json_utils import NumpyEncoder  
from .utils import logging_utils 
from .config import SAMPLE_RATE, SERVER_IP, SERVER_PORT 

logger = logging_utils.get_logger(__name__)

STATE_EMOJI = {
    "VOICE_RECEIVED": "🟢",
    "STT_DIARIZATION": "🔵",
    "LLM": "🟣",
    "TTS": "🟠",
    "WAKEWORD": "🔴",
    "VAD": "🟢",
}

class Server:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.server_socket = None
        self.client_socket = None
        self.client_address = None
        self.audio_processor = AudioProcessor()
        self.llm_handler = LLMHandler()
        self.tts_handler = TTSHandler()
        self.conversation_stats_list = []

        logger.info("Server initialized.")

    def _start_server(self):
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.bind((self.ip, self.port))
            self.server_socket.listen(1)
            msg = f"Server started on {self.ip}:{self.port}. Waiting for connections..."
            logger.info(msg)
        except Exception as e:
            logger.exception(f"Failed to start server: {e}")
            raise

    def _accept_client(self):
        try:
            self.client_socket, self.client_address = self.server_socket.accept()
            msg = f"Connection from {self.client_address} established."
            logger.info(msg)
        except Exception as e:
            logger.exception(f"Failed to accept client connection: {e}")
            raise

    def _receive_audio(self) -> Optional[np.ndarray]:
        """
        Receives an int32 size header followed by that many bytes of float32 audio.
        Returns None on connection close or error.
        """
        try:
            data_size_bytes = self.client_socket.recv(4)
            if not data_size_bytes:
                logger.info("Client closed the connection.")
                return None
            data_size = struct.unpack("!I", data_size_bytes)[0]
            msg = f"Receiving audio data of size: {data_size} bytes"
            logger.info(msg)

            audio_data = b""
            bytes_received = 0
            while bytes_received < data_size:
                chunk = self.client_socket.recv(min(4096, data_size - bytes_received))
                if not chunk:
                    msg = "Client closed connection during audio reception."
                    logger.info(msg)
                    return None
                audio_data += chunk
                bytes_received += len(chunk)

            msg = f"🟢 Received {bytes_received} bytes of audio data."
            logger.info(msg)
            audio_np = np.frombuffer(audio_data, dtype=np.float32)
            logger.debug(f"Audio data shape: {audio_np.shape}")
            return audio_np

        except Exception as e:
            logger.exception(f"Error receiving audio: {e}")
            return None

    def _send_response(self, audio_data, next_state):
        """
        Sends a JSON response with base64-encoded audio and the next state.
        Handles potential errors during sending.
        """
        try:
            if audio_data is not None:
                audio_base64 = base64.b64encode(audio_data.tobytes()).decode('utf-8')
            else:
                audio_base64 = ""

            response_data = {
                "audio": audio_base64,
                "next_state": next_state
            }
            response_json = json.dumps(response_data, cls=NumpyEncoder)
            response_size = len(response_json)

            logger.debug(f"Sending response of size: {response_size} bytes")
            self.client_socket.sendall(struct.pack("!I", response_size))
            self.client_socket.sendall(response_json.encode())

            emoji = STATE_EMOJI.get(next_state, "🔵")
            msg = f"Response sent (audio len={len(audio_base64)}) with next state '{emoji} {next_state}'."
            logger.info(msg)
            print(msg)
        except Exception as e:
            logger.exception(f"Error sending response: {e}")

    def run(self):
        self._start_server()
        try:
            while True:
                self._accept_client()
                while True:
                    audio_data = self._receive_audio()
                    if audio_data is None:
                        break  # Client disconnected

                    audio_received_bytes = audio_data.nbytes if audio_data is not None else 0
                    processing_start = time.perf_counter()

                    # 1. Diarization and STT
                    t_stt_start = time.perf_counter()
                    conversation_text, dia_stats = self.audio_processor.perform_diarization_stt(
                        audio_data, SAMPLE_RATE
                    )
                    t_stt = (time.perf_counter() - t_stt_start) * 1000
                    msg_stt = f"🔵 Diarization + STT completed in {t_stt:.2f} ms."
                    logger.info(msg_stt)
                    logger.debug(f"Diarization + STT result:\n{dia_stats}")
                    print("Transcript:\n", conversation_text)

                    # 2. LLM
                    t_llm_start = time.perf_counter()
                    response, probability = self.llm_handler.process_input(conversation_text)
                    t_llm = (time.perf_counter() - t_llm_start) * 1000
                    msg_llm = f"🟣 LLM response (took {t_llm:.2f} ms):\n{response}"
                    logger.info(msg_llm)
                    token_count = len(self.llm_handler.tokenizer.encode(response)) if hasattr(self.llm_handler, 'tokenizer') and self.llm_handler.tokenizer is not None else 0

                    # 3. TTS
                    t_tts_start = time.perf_counter()
                    audio_response, tts_meta = self.tts_handler.generate_tts(response)
                    t_tts = (time.perf_counter() - t_tts_start) * 1000
                    msg_tts = f"🟠 TTS generation completed in {t_tts:.2f} ms."
                    logger.info(msg_tts)
                    print(msg_tts)
                    if tts_meta:
                        logger.debug(f"TTS phonemes: {tts_meta}")

                    # 4. Decide next state
                    next_state = "WAKEWORD" if probability > 0.5 else "VAD"

                    # 5. Send Response
                    self._send_response(audio_response, next_state)
                    processing_elapsed = (time.perf_counter() - processing_start) * 1000

                    # Create conversation stats
                    conversation_stats = {
                        "timestamp": time.time(),
                        "audio_received_bytes": audio_received_bytes,
                        "stt_diarization_ms": t_stt,
                        "stt_characters": len(conversation_text),
                        "llm_ms": t_llm,
                        "llm_token_count": token_count,
                        "tts_ms": t_tts,
                        "tts_phonemes": tts_meta,
                        "total_processing_ms": processing_elapsed,
                        "diarization": dia_stats
                    }
                    self.conversation_stats_list.append(conversation_stats)
                    stats_str = json.dumps(conversation_stats, indent=2, cls=NumpyEncoder)
                    logger.debug("Conversation stats:\n" + stats_str)
                    logger.info(f"Total processing time: {processing_elapsed:.2f} ms.")

        except KeyboardInterrupt:
            logger.info("Stopping server (KeyboardInterrupt).")
            print("Stopping server (KeyboardInterrupt).")
        finally:
            self.shutdown()
            self.save_stats()

    def shutdown(self):
        logger.info("Shutting down server...")
        try:
            if self.client_socket:
                self.client_socket.close()
        except Exception as e:
            logger.error(f"Error closing client socket: {e}")
        try:
            if self.server_socket:
                self.server_socket.close()
        except Exception as e:
            logger.error(f"Error closing server socket: {e}")
        logger.info("Server shut down.")
        print("Server shut down.")

    def save_stats(self):
        stats_filename = "conversation_stats.json"
        try:
            existing_stats = []
            if os.path.exists(stats_filename):
                try:
                    with open(stats_filename, "r") as f:
                        existing_stats = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not read existing stats: {e}")
            existing_stats.extend(self.conversation_stats_list)
            with open(stats_filename, "w") as f:
                json.dump(existing_stats, f, indent=2, cls=NumpyEncoder)
            msg = f"Stats appended to {stats_filename}."
            print(msg)
        except Exception as e:
            logger.error(f"Failed to save conversation stats: {e}")
