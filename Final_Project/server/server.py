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

# Import common speaker embedding utilities
from utils import (
    load_speaker_database,
    extract_sliding_embedding_for_segment,
    speaker_id_from_embedding
)

# Custom JSON Encoder for NumPy Types (if needed)
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
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
# Load Models and Pipelines
# ------------------------------------------------------------------------------
try:
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN
    )
    logger.info("Diarization pipeline loaded successfully.")
except Exception as e:
    logger.warning(f"Could not load pyannote/speaker-diarization-3.1: {e}")
    diarization_pipeline = None

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
        skip_aggregation=True,  # We'll do manual aggregation via our utils
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
    logger.warning("Resegmentation module not available.")

def resegment_with_embeddings(diar_result, wav_path):
    """
    If available, run embedding-based re-segmentation to refine diarization boundaries.
    """
    if resegmenter is None:
        logger.debug("No resegmenter available. Skipping re-segmentation.")
        return diar_result
    logger.debug("Performing embedding-based re-segmentation...")
    return resegmenter(wav_path, diar_result)

def merge_short_segments(segments, min_duration_merge=0.7):
    """
    Merge consecutive segments by the same speaker if a segment is shorter than `min_duration_merge`
    or if there's a tiny gap between them.
    """
    if not segments:
        return []
    merged_segments = []
    prev = segments[0]
    for current in segments[1:]:
        duration = current["end"] - current["start"]
        if current["speaker"] == prev["speaker"] and duration < min_duration_merge:
            prev["end"] = current["end"]
            prev["stt_text"] = prev["stt_text"].strip() + " " + current["stt_text"].strip()
            prev["similarity"] = max(prev["similarity"], current["similarity"])
        else:
            merged_segments.append(prev)
            prev = current
    merged_segments.append(prev)
    return merged_segments

# ------------------------------------------------------------------------------
# Conversation State and LLM Handler
# ------------------------------------------------------------------------------
@dataclass
class ConversationState:
    context: List[dict]
    last_response: Optional[str] = None

class LLamaConversationHandler:
    def __init__(self, model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
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

        self.conversation = ConversationState(context=[])
        logger.info("LLama conversation handler initialized.")

    def process_input(self, text: str) -> Tuple[str, float]:
        # Append user input to conversation context.
        self.conversation.context.append({"role": "user", "content": text})
        
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant on a home speaker. Respond concisely without repeating entire logs. You'll hear multiple people with names if available. Respond to the speakers."
            }
        ] + self.conversation.context

        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = ""
            for msg in messages:
                prompt += f"<{msg['role']}>: {msg['content']}\n"
        
        logger.debug(f"LLM Prompt:\n{prompt}")
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
        self.conversation.context.append({"role": "assistant", "content": assistant_reply})
        probability = 0.9 if any(kw in text.lower() for kw in ["bye", "goodbye", "stop", "end"]) else 0.0
        return assistant_reply, probability

# ------------------------------------------------------------------------------
# Kokoro TTS Functions
# ------------------------------------------------------------------------------
def build_kokoro_tts():
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    tts_model = build_model("../datasets/Kokoro-82M/kokoro-v0_19.pth", device_str)
    voice_obj = torch.load("../datasets/Kokoro-82M/voices/af.pt", weights_only=True).to(device_str)
    logger.info("Kokoro TTS model and voice loaded.")
    return tts_model, voice_obj

def generate_tts(tts_model, voice, text) -> Tuple[Optional[np.ndarray], Optional[str]]:
    logger.info("🟠 Generating TTS audio with Kokoro...")
    try:
        start_time = time.perf_counter()
        audio, out_ps = generate(tts_model, text, voice, lang="a")
        elapsed = (time.perf_counter() - start_time) * 1000
        logger.info(f"🟠 Phonemes generated: {out_ps} (TTS took {elapsed:.2f} ms)")
        audio_np = audio.astype(np.float32)
        audio_norm = audio_np / np.iinfo(np.int16).max
        return audio_norm, out_ps
    except Exception as e:
        logger.error(f"Error in TTS generation: {e}")
        return None, None

# ------------------------------------------------------------------------------
# Diarization and STT Processing
# ------------------------------------------------------------------------------
def perform_diarization_stt(
    audio_data: np.ndarray,
    sample_rate: int,
    stt_pipeline,
    speaker_db: dict
) -> Tuple[str, Dict[str, Any]]:
    overall_start = time.perf_counter()

    segments_stats: List[Dict[str, Any]] = []

    # Fallback if diarization or embedding not loaded.
    if diarization_pipeline is None or embedding_inference is None:
        logger.warning("Diarization or embedding model not loaded. Using entire-file STT only.")
        stt_start = time.perf_counter()
        result = stt_pipeline({"raw": audio_data, "sampling_rate": sample_rate}, chunk_length_s=30)
        stt_elapsed = (time.perf_counter() - stt_start) * 1000
        logger.info(f"🔵 [STT-Fallback] Transcribed text: {result['text']} (STT took {stt_elapsed:.2f} ms)")
        dia_stats = {
            "segments": [{
                "start": None,
                "end": None,
                "speaker": "Fallback",
                "similarity": None,
                "stt_text": result["text"],
                "rms": None
            }],
            "total_segments": 1,
            "unique_speakers": ["Fallback"],
            "total_stt_characters": len(result["text"])
        }
        total_elapsed = (time.perf_counter() - overall_start) * 1000
        logger.info(f"🔵 Diarization+STT completed in {total_elapsed:.2f} ms.")
        return result["text"], dia_stats

    # Save temporary WAV file for processing.
    temp_wav = "temp_received.wav"
    with wave.open(temp_wav, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        int16_data = (audio_data * 32767).astype(np.int16)
        wf.writeframes(int16_data.tobytes())

    file_duration = len(audio_data) / sample_rate

    # 1) Initial Diarization
    diag_start = time.perf_counter()
    diarization = diarization_pipeline(temp_wav)
    diag_elapsed = (time.perf_counter() - diag_start) * 1000
    logger.info(f"🔵 Diarization completed in {diag_elapsed:.2f} ms.")

    # 2) Embedding-Based Re-Segmentation (if available)
    diarization = resegment_with_embeddings(diarization, temp_wav)

    # Build raw segments
    raw_segments = []
    for turn, _, label in diarization.itertracks(yield_label=True):
        start_time_seg = max(0.0, min(turn.start, file_duration))
        end_time_seg = max(0.0, min(turn.end, file_duration))
        if end_time_seg > start_time_seg:
            raw_segments.append((start_time_seg, end_time_seg, label))
    raw_segments.sort(key=lambda x: x[0])

    # 3) For each segment, extract speaker embedding & run STT
    for (start_time_seg, end_time_seg, diar_label) in raw_segments:
        start_idx = int(start_time_seg * sample_rate)
        end_idx = int(end_time_seg * sample_rate)
        segment_samples = audio_data[start_idx:end_idx]
        rms = np.sqrt(np.mean(segment_samples ** 2))

        seg_emb = extract_sliding_embedding_for_segment(temp_wav, start_time_seg, end_time_seg, embedding_inference)
        spk_id, similarity = speaker_id_from_embedding(seg_emb, speaker_db, threshold=0.25)

        # Run STT on the segment
        stt_seg_start = time.perf_counter()
        stt_result = stt_pipeline({"raw": segment_samples, "sampling_rate": sample_rate}, chunk_length_s=30)
        stt_seg_elapsed = (time.perf_counter() - stt_seg_start) * 1000
        text = stt_result["text"]

        logger.info(f"🔵 [Segment STT] {spk_id} => {text} (Segment STT took {stt_seg_elapsed:.2f} ms)")

        segments_stats.append({
            "start": start_time_seg,
            "end": end_time_seg,
            "speaker": spk_id,
            "similarity": similarity,
            "stt_text": text,
            "rms": rms,
            "segment_stt_ms": stt_seg_elapsed
        })

    # 4) Merge or Relabel Short Segments
    merged_segments = merge_short_segments(segments_stats, min_duration_merge=0.7)
    final_text = "\n".join(f"{seg['speaker']} said: {seg['stt_text']}" for seg in merged_segments)
    total_stt_characters = sum(len(seg["stt_text"]) for seg in merged_segments)
    unique_speakers = list({seg["speaker"] for seg in merged_segments})

    # Compute average similarity per speaker (if needed)
    speaker_similarities: Dict[str, List[float]] = {}
    for seg in merged_segments:
        speaker_similarities.setdefault(seg["speaker"], []).append(seg["similarity"])
    avg_speaker_similarities = {spk: float(np.mean(sims)) for spk, sims in speaker_similarities.items()}

    dia_stats = {
        "segments": merged_segments,
        "total_segments": len(merged_segments),
        "unique_speakers": unique_speakers,
        "total_stt_characters": total_stt_characters,
        "speaker_similarities": avg_speaker_similarities
    }
    total_elapsed = (time.perf_counter() - overall_start) * 1000
    logger.info(f"🔵 Diarization transcript:\n{final_text}\n(Total diarization+STT took {total_elapsed:.2f} ms)")
    return final_text, dia_stats

# ------------------------------------------------------------------------------
# Main Server Class
# ------------------------------------------------------------------------------
class Server:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.server_socket = None
        self.client_socket = None
        self.client_address = None

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.stt_pipeline = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3-turbo",
            device=self.device
        )
        self.llm_handler = LLamaConversationHandler(
            model_id="meta-llama/Meta-Llama-3-8B-Instruct"
        )
        self.tts_model, self.voice = build_kokoro_tts()

        # Load speaker database (using relative path if needed)
        self.speaker_db = load_speaker_database("speaker_embeddings.json")
        logger.info("Server initialized.")

    def _start_server(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.ip, self.port))
        self.server_socket.listen(1)
        logger.info(f"Server started on {self.ip}:{self.port}. Waiting for connection...")

    def _accept_client(self):
        self.client_socket, self.client_address = self.server_socket.accept()
        logger.info(f"Connection from {self.client_address} established.")

    def _receive_audio(self) -> Optional[np.ndarray]:
        data_size_bytes = self.client_socket.recv(4)
        if not data_size_bytes:
            logger.info("Client closed the connection.")
            return None
        data_size = struct.unpack("!I", data_size_bytes)[0]
        logger.info(f"Receiving audio data of size: {data_size} bytes")
        audio_data = b""
        bytes_received = 0
        while bytes_received < data_size:
            chunk = self.client_socket.recv(min(4096, data_size - bytes_received))
            if not chunk:
                logger.info("Client closed connection during audio reception.")
                return None
            audio_data += chunk
            bytes_received += len(chunk)
        logger.info(f"🟢 Received {bytes_received} bytes of audio data.")
        audio_np = np.frombuffer(audio_data, dtype=np.float32)
        logger.debug(f"Audio data shape: {audio_np.shape}")
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
        response_json = json.dumps(response_data, cls=NumpyEncoder)
        response_size = len(response_json)
        logger.debug(f"Sending response of size: {response_size} bytes")
        self.client_socket.sendall(struct.pack("!I", response_size))
        self.client_socket.sendall(response_json.encode())
        state_emoji = STATE_EMOJI.get(next_state, "🔵")
        logger.info(f"Response sent with audio size {len(audio_base64)} bytes and next state '{state_emoji} {next_state}'.")

    def run(self):
        self._start_server()
        try:
            while True:
                self._accept_client()
                while True:
                    audio_data = self._receive_audio()
                    if audio_data is None:
                        break  # Client disconnected

                    audio_received_bytes = audio_data.nbytes
                    processing_start = time.perf_counter()

                    # Diarization + STT processing
                    t_stt_start = time.perf_counter()
                    conversation_text, dia_stats = perform_diarization_stt(
                        audio_data,
                        sample_rate=SAMPLE_RATE,
                        stt_pipeline=self.stt_pipeline,
                        speaker_db=self.speaker_db
                    )
                    t_stt = (time.perf_counter() - t_stt_start) * 1000
                    logger.info(f"🔵 STT & Diarization completed in {t_stt:.2f} ms.")

                    # Process conversation via LLM
                    t_llm_start = time.perf_counter()
                    logger.info("🟣 Interacting with LLM...")
                    response, probability = self.llm_handler.process_input(conversation_text)
                    t_llm = (time.perf_counter() - t_llm_start) * 1000
                    logger.info(f"🟣 LLM response (took {t_llm:.2f} ms):\n------------------\n{response}\n------------------")
                    token_count = len(self.llm_handler.tokenizer.encode(response))

                    # Generate TTS audio
                    t_tts_start = time.perf_counter()
                    audio_response, tts_phonemes = generate_tts(self.tts_model, self.voice, response)
                    t_tts = (time.perf_counter() - t_tts_start) * 1000
                    logger.info(f"🟠 TTS generation completed in {t_tts:.2f} ms.")

                    # Decide next state (e.g., WAKEWORD vs VAD)
                    next_state = "WAKEWORD" if probability > 0.5 else "VAD"

                    processing_elapsed = (time.perf_counter() - processing_start) * 1000
                    conversation_stats = {
                        "timestamp": time.time(),
                        "audio_received_bytes": audio_received_bytes,
                        "stt_diarization_ms": t_stt,
                        "stt_characters": len(conversation_text),
                        "llm_ms": t_llm,
                        "llm_token_count": token_count,
                        "tts_ms": t_tts,
                        "tts_phonemes": tts_phonemes,
                        "total_processing_ms": processing_elapsed,
                        "diarization": dia_stats
                    }
                    conversation_stats_list.append(conversation_stats)
                    logger.debug(f"Conversation stats: {json.dumps(conversation_stats, indent=2, cls=NumpyEncoder)}")

                    self._send_response(audio_response, next_state)
                    logger.info(f"Total processing time for this cycle: {processing_elapsed:.2f} ms.\n")

        except KeyboardInterrupt:
            logger.info("Stopping server due to keyboard interrupt...")
        finally:
            if self.client_socket:
                self.client_socket.close()
            if self.server_socket:
                self.server_socket.close()
            logger.info("Server shut down.")

            # Save conversation stats
            stats_filename = "conversation_stats.json"
            existing_stats = []
            if os.path.exists(stats_filename):
                try:
                    with open(stats_filename, "r") as f:
                        existing_stats = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not read existing stats file: {e}")
            existing_stats.extend(conversation_stats_list)
            with open(stats_filename, "w") as f:
                json.dump(existing_stats, f, indent=2, cls=NumpyEncoder)
            logger.info(f"Conversation statistics appended to '{stats_filename}'.")



if __name__ == "__main__":
    server = Server(SERVER_IP, SERVER_PORT)
    server.run()
