# server.py
import socket
import struct
import numpy as np
import torch
import sys
import os
import json
import base64
import wave
import time
import logging
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any

# HUGGING FACE imports
from transformers import pipeline

# pyannote
from pyannote.audio import Pipeline
from pyannote.audio import Model as EmbeddingModel
from pyannote.audio import Inference
from pyannote.core import Segment

# Re-segmentation
try:
    from pyannote.audio.pipelines.utils.resegmentation import Resegmentation
except ImportError:
    Resegmentation = None

# Kokoro TTS imports (adapt paths as needed)
sys.path.append(os.path.abspath("../datasets/Kokoro-82M"))
from models import build_model
from kokoro import generate

from dotenv import load_dotenv
load_dotenv()

# Import our utils
from utils import (
    load_speaker_database,
    extract_sliding_embedding_for_segment,
    speaker_id_from_embedding
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

STATE_EMOJI = {
    "VOICE_RECEIVED": "🟢",
    "STT_DIARIZATION": "🔵",
    "LLM": "🟣",
    "TTS": "🟠",
    "WAKEWORD": "🔴",
    "VAD": "🟢",
}

conversation_stats_list: List[Dict[str, Any]] = []

SERVER_IP = "0.0.0.0"
SERVER_PORT = 8080
SAMPLE_RATE = 16000
HF_TOKEN = os.getenv("HF_AUTH_TOKEN", None)


# ------------------------------------------------------------------------------
# Diarization pipeline, Re-seg
# ------------------------------------------------------------------------------
try:
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN
    )
    logger.info("Diarization pipeline loaded.")
except Exception as e:
    logger.warning(f"Could not load diarization pipeline: {e}")
    diarization_pipeline = None

# Attempt to load embedding model in "sliding" mode
try:
    embedding_model = EmbeddingModel.from_pretrained(
        "pyannote/embedding",
        token=HF_TOKEN
    )
    logger.info("Embedding model loaded successfully.")
except Exception as e:
    logger.warning(f"Could not load pyannote/embedding: {e}")
    embedding_model = None

if embedding_model is not None:
    embedding_inference = Inference(
        embedding_model,
        skip_aggregation=True,
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        window="sliding",
        duration=1.5,
        step=0.75
    )
else:
    embedding_inference = None

if Resegmentation and embedding_model:
    resegmenter = Resegmentation(segmentation=embedding_model)
    logger.info("Resegmentation module instantiated.")
else:
    resegmenter = None
    logger.warning("No Resegmentation available.")


def resegment_with_embeddings(diar_result, wav_path):
    if resegmenter is None:
        return diar_result
    logger.debug("Performing embedding-based re-segmentation.")
    return resegmenter(wav_path, diar_result)


def merge_short_segments(segments, min_duration_merge=0.7):
    """
    Merge consecutive segments by the same speaker if the segment is shorter than `min_duration_merge`
    or there's a small gap, etc.
    """
    if not segments:
        return segments

    merged_segments = []
    prev = segments[0]
    for current in segments[1:]:
        duration = current["end"] - current["start"]
        # If same speaker + short segment => merge
        if current["speaker"] == prev["speaker"] and duration < min_duration_merge:
            prev["end"] = current["end"]
            prev["stt_text"] += " " + current["stt_text"]
            prev["similarity"] = max(prev["similarity"], current["similarity"])
        else:
            merged_segments.append(prev)
            prev = current
    merged_segments.append(prev)
    return merged_segments


# ------------------------------------------------------------------------------
# STT Pipeline
# ------------------------------------------------------------------------------
class LLamaConversationHandler:
    def __init__(self, model_id="meta-llama/Meta-Llama-3-8B-Instruct"):
        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            device_map="auto",
            model_kwargs={"torch_dtype": torch.bfloat16},
        )
        self.tokenizer = self.pipe.tokenizer

        eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot_id == self.tokenizer.unk_token_id:
            self.terminators = [self.tokenizer.eos_token_id]
        else:
            self.terminators = [self.tokenizer.eos_token_id, eot_id]

        self.conversation = []
        logger.info("LLama conversation handler initialized.")

    def process_input(self, text: str):
        # minimal example, append user input
        self.conversation.append({"role": "user", "content": text})

        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant. You may see multiple speakers. Respond accordingly."
            }
        ] + self.conversation

        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            prompt += f"<{role}>: {content}\n"

        logger.debug(f"LLM prompt:\n{prompt}")
        output = self.pipe(
            prompt,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            eos_token_id=self.terminators
        )
        full_text = output[0]["generated_text"]
        assistant_reply = full_text[len(prompt):].strip()

        # Append assistant reply
        self.conversation.append({"role": "assistant", "content": assistant_reply})

        # trivial approach to "probability"
        probability = 0.0
        if any(kw in text.lower() for kw in ["bye", "goodbye", "stop"]):
            probability = 0.9
        return assistant_reply, probability


def build_kokoro_tts():
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "../datasets/Kokoro-82M/kokoro-v0_19.pth"
    voice_path = "../datasets/Kokoro-82M/voices/af.pt"

    tts_model = build_model(model_path, device_str)
    voice_obj = torch.load(voice_path, weights_only=True).to(device_str)
    logger.info("Kokoro TTS model and voice loaded.")
    return tts_model, voice_obj

def generate_tts(tts_model, voice, text):
    logger.info("Generating TTS with Kokoro...")
    try:
        start_time = time.perf_counter()
        audio, out_ps = generate(tts_model, text, voice, lang="a")
        elapsed = (time.perf_counter() - start_time) * 1000
        logger.info(f"TTS took {elapsed:.2f} ms.")
        audio_np = audio.astype(np.float32) / np.iinfo(np.int16).max
        return audio_np, out_ps
    except Exception as e:
        logger.error(f"Error in TTS generation: {e}")
        return None, None


def perform_diarization_stt(
    audio_data: np.ndarray,
    sample_rate: int,
    stt_pipeline,
    speaker_db: dict
):
    """
    Diarize -> resegment -> identify each speaker -> STT each segment.
    Return final text, plus metadata in 'dia_stats'.
    """
    overall_start = time.perf_counter()
    if diarization_pipeline is None or embedding_inference is None:
        logger.warning("Diarization or embedding not loaded. Doing entire-file STT only.")
        result = stt_pipeline({"raw": audio_data, "sampling_rate": sample_rate})
        return result["text"], {"fallback": True}

    # Save to temp WAV
    temp_wav = "temp_received.wav"
    with wave.open(temp_wav, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

    diar_start = time.perf_counter()
    diar = diarization_pipeline(temp_wav)
    diar_elapsed = (time.perf_counter() - diar_start)*1000
    logger.info(f"Diarization took {diar_elapsed:.2f} ms.")

    diar = resegment_with_embeddings(diar, temp_wav)

    # Build segments
    file_duration = len(audio_data)/sample_rate
    segments_list = []
    for turn, _, spk_label in diar.itertracks(yield_label=True):
        stt = max(0.0, turn.start)
        et = min(file_duration, turn.end)
        if et > stt:
            segments_list.append((stt, et, spk_label))
    segments_list.sort(key=lambda x: x[0])

    # Identify & STT
    final_segments = []
    for (start_time_seg, end_time_seg, label) in segments_list:
        seg_audio = audio_data[int(start_time_seg*sample_rate):int(end_time_seg*sample_rate)]
        rms = np.sqrt(np.mean(seg_audio**2))
        if rms < 0.01:
            spk_id, sim = ("Unknown", 0.0)
            text = ""
        else:
            seg_emb = extract_sliding_embedding_for_segment(
                temp_wav, start_time_seg, end_time_seg, embedding_inference
            )
            spk_id, sim = speaker_id_from_embedding(seg_emb, speaker_db, threshold=0.25)

            stt_result = stt_pipeline({"raw": seg_audio, "sampling_rate": sample_rate})
            text = stt_result["text"]

        final_segments.append({
            "start": start_time_seg,
            "end": end_time_seg,
            "speaker": spk_id,
            "similarity": sim,
            "stt_text": text
        })

    merged = merge_short_segments(final_segments, 0.7)
    final_text = "\n".join(f"{seg['speaker']} said: {seg['stt_text']}" for seg in merged)
    total_elapsed = (time.perf_counter()-overall_start)*1000

    dia_stats = {
        "segments": merged,
        "total_segments": len(merged),
        "total_processing_ms": total_elapsed
    }

    logger.info(f"Diarization+STT completed in {total_elapsed:.2f} ms.")
    return final_text, dia_stats


class Server:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.server_socket = None
        self.client_socket = None
        self.client_address = None

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.stt_pipeline = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3-turbo",
            device=self.device
        )
        self.llm_handler = LLamaConversationHandler()
        self.tts_model, self.voice = build_kokoro_tts()

        # Load DB
        self.speaker_db = load_speaker_database("speaker_embeddings.json")

    def _start_server(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.ip, self.port))
        self.server_socket.listen(1)
        logger.info(f"Server started on {self.ip}:{self.port}. Waiting for connection...")

    def _accept_client(self):
        self.client_socket, self.client_address = self.server_socket.accept()
        logger.info(f"Connection from {self.client_address} established.")

    def _receive_audio(self) -> Optional[np.ndarray]:
        size_bytes = self.client_socket.recv(4)
        if not size_bytes:
            return None
        data_size = struct.unpack("!I", size_bytes)[0]
        audio_data = b""
        bytes_received = 0
        while bytes_received < data_size:
            chunk = self.client_socket.recv(min(4096, data_size - bytes_received))
            if not chunk:
                return None
            audio_data += chunk
            bytes_received += len(chunk)

        logger.info(f"Received {bytes_received} bytes of audio data.")
        audio_np = np.frombuffer(audio_data, dtype=np.float32)
        return audio_np

    def _send_response(self, audio_data, next_state):
        if audio_data is not None:
            audio_base64 = base64.b64encode(audio_data.tobytes()).decode('utf-8')
        else:
            audio_base64 = ""

        response_data = {
            "audio": audio_base64,
            "next_state": next_state
        }
        response_json = json.dumps(response_data)
        response_size = len(response_json)

        self.client_socket.sendall(struct.pack("!I", response_size))
        self.client_socket.sendall(response_json.encode())

        state_emoji = STATE_EMOJI.get(next_state, "🔵")
        logger.info(f"Sent response, next_state='{state_emoji} {next_state}'.")

    def run(self):
        self._start_server()
        try:
            while True:
                self._accept_client()
                while True:
                    audio_data = self._receive_audio()
                    if audio_data is None:
                        logger.info("Client disconnected.")
                        break

                    # Diarization + STT
                    conversation_text, dia_stats = perform_diarization_stt(
                        audio_data, SAMPLE_RATE, self.stt_pipeline, self.speaker_db
                    )

                    # LLM
                    llm_reply, probability = self.llm_handler.process_input(conversation_text)
                    logger.info(f"LLM reply:\n{llm_reply}")

                    # TTS
                    audio_response, phonemes = generate_tts(self.tts_model, self.voice, llm_reply)

                    # Decide next state
                    next_state = "WAKEWORD" if probability > 0.5 else "VAD"

                    self._send_response(audio_response, next_state)

        except KeyboardInterrupt:
            logger.info("Server stopping via keyboard interrupt.")
        finally:
            if self.client_socket:
                self.client_socket.close()
            if self.server_socket:
                self.server_socket.close()
            logger.info("Server shut down.")

if __name__ == "__main__":
    server = Server(SERVER_IP, SERVER_PORT)
    server.run()
