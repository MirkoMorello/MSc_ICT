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

# Kokoro TTS imports
sys.path.append(os.path.abspath("../datasets/Kokoro-82M"))
from models import build_model
from kokoro import generate

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# Custom JSON Encoder for NumPy Types
# =============================================================================

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# =============================================================================
# Configuration and Logging Setup
# =============================================================================

SERVER_IP = "0.0.0.0"
SERVER_PORT = 8080
SAMPLE_RATE = 16000
DEBUG = True
MODEL_PATH = "../datasets/Kokoro-82M/kokoro-v0_19.pth"
VOICE_PATH = "../datasets/Kokoro-82M/voices/af.pt"
VOICE_NAME = "af"

HF_TOKEN = os.getenv("HF_AUTH_TOKEN", None)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Define emoji mapping for different states/steps
STATE_EMOJI = {
    "VOICE_RECEIVED": "🟢",
    "STT_DIARIZATION": "🔵",
    "LLM": "🟣",
    "TTS": "🟠",
    "WAKEWORD": "🔴",
    "VAD": "🟢",
}

# Global list to store conversation statistics
conversation_stats_list: List[Dict[str, Any]] = []

# =============================================================================
# Load Models and Pipelines
# =============================================================================

# Attempt to load pyannote diarization pipeline
try:
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1"
        # If it is private and you do need auth, use: use_auth_token=HF_TOKEN
    )
    logger.info("Diarization pipeline loaded successfully.")
except Exception as e:
    logger.warning(f"Could not load pyannote/speaker-diarization-3.1: {e}")
    diarization_pipeline = None

# Attempt to load embedding model
try:
    embedding_model = EmbeddingModel.from_pretrained(
        "pyannote/embedding"
        # If private, you might need: use_auth_token=HF_TOKEN
    )
    logger.info("Embedding model loaded successfully.")
except Exception as e:
    logger.warning(f"Could not load pyannote/embedding: {e}")
    embedding_model = None

if embedding_model is not None:
    embedding_inference = Inference(
        embedding_model,
        skip_aggregation=True,
        device=device
    )
else:
    embedding_inference = None

# =============================================================================
# Conversation State and LLM Handler
# =============================================================================

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
        # Append user input to conversation history.
        self.conversation.context.append({"role": "user", "content": text})
        
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant on a home speaker. Respond concisely without repeating entire logs. You'll hear multiple people with names if available. Respond to the speakers."}
        ]
        messages.extend(self.conversation.context)

        # Build prompt using tokenizer's chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                prompt += f"<{role}>: {content}\n"

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

        probability = 0.0
        if any(kw in text.lower() for kw in ["bye", "goodbye", "stop", "end"]):
            probability = 0.9

        self.conversation.context.append({"role": "assistant", "content": assistant_reply})
        return assistant_reply, probability

# =============================================================================
# Kokoro TTS Functions
# =============================================================================

def build_kokoro_tts():
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    tts_model = build_model(MODEL_PATH, device_str)
    voice_obj = torch.load(VOICE_PATH, weights_only=True).to(device_str)
    logger.info("Kokoro TTS model and voice loaded.")
    return tts_model, voice_obj

def generate_tts(tts_model, voice, text) -> Tuple[Optional[np.ndarray], Optional[str]]:
    logger.info("🟠 Generating TTS audio with Kokoro...")
    try:
        start_time = time.perf_counter()
        audio, out_ps = generate(tts_model, text, voice, lang=VOICE_NAME[0])
        elapsed = (time.perf_counter() - start_time) * 1000
        logger.info(f"🟠 Phonemes generated: {out_ps} (TTS took {elapsed:.2f} ms)")
        audio_np = audio
        audio_norm = audio_np.astype(np.float32) / np.iinfo(np.int16).max
        return audio_norm, out_ps
    except Exception as e:
        logger.error(f"Error in TTS generation: {e}")
        return None, None

# =============================================================================
# Speaker Database and Identification
# =============================================================================

def load_speaker_database(json_path="speaker_embeddings.json"):
    if not os.path.isfile(json_path):
        logger.warning(f"Speaker database not found at {json_path}.")
        return {}
    with open(json_path, "r") as f:
        logger.info(f"Loading speaker database from {json_path}")
        return json.load(f)

def speaker_id_from_embedding(embedding_vector: np.ndarray, speaker_db: dict, threshold=0.7) -> Tuple[str, float]:
    if not speaker_db:
        return "Unknown", 0.0

    if embedding_vector.ndim > 1:
        embedding_vector = np.mean(embedding_vector, axis=0)

    emb = embedding_vector.squeeze()
    emb_norm = np.linalg.norm(emb) + 1e-10
    emb_normed = emb / emb_norm

    logger.debug(f"[DEBUG] Inference embedding shape: {embedding_vector.shape}, norm: {np.linalg.norm(emb):.4f}")

    best_spk = "Unknown"
    best_score = -1.0

    logger.debug("Similarity to each speaker in DB:")
    for spk_name, centroid in speaker_db.items():
        centroid_arr = np.array(centroid)
        if centroid_arr.ndim > 1:
            centroid_arr = np.mean(centroid_arr, axis=0)
        centroid_arr = centroid_arr.squeeze()
        centroid_norm = np.linalg.norm(centroid_arr) + 1e-10
        centroid_normed = centroid_arr / centroid_norm

        sim = np.dot(emb_normed, centroid_normed)
        logger.debug(f"  {spk_name}: cos_sim={sim:.4f}, centroid_norm={centroid_norm:.4f}")

        if sim > best_score:
            best_score = sim
            best_spk = spk_name

    logger.debug(f"[DEBUG] Best speaker = {best_spk}, similarity = {best_score:.4f}, threshold={threshold}")

    if best_score < threshold:
        logger.debug("→ Best score below threshold. Returning 'Unknown'.")
        return "Unknown", best_score

    logger.debug("→ Recognized speaker above threshold.")
    return best_spk, best_score

# =============================================================================
# Diarization and STT Processing
# =============================================================================

def perform_diarization_stt(
    audio_data: np.ndarray,
    sample_rate: int,
    stt_pipeline,
    speaker_db: dict
) -> Tuple[str, Dict[str, Any]]:
    overall_start = time.perf_counter()

    segments_stats: List[Dict[str, Any]] = []

    if diarization_pipeline is None or embedding_inference is None:
        logger.warning("Diarization or embedding model not loaded. Using entire STT only.")
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

    # Save temporary WAV file for pyannote processing.
    temp_wav = "temp_received.wav"
    with wave.open(temp_wav, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        int16_data = (audio_data * 32767).astype(np.int16)
        wf.writeframes(int16_data.tobytes())

    file_duration = len(audio_data) / sample_rate
    diag_start = time.perf_counter()
    diarization = diarization_pipeline(temp_wav)
    diag_elapsed = (time.perf_counter() - diag_start) * 1000
    logger.info(f"🔵 Diarization completed in {diag_elapsed:.2f} ms.")

    for turn, _, _ in diarization.itertracks(yield_label=True):
        start_time_seg = max(0.0, min(turn.start, file_duration))
        end_time_seg   = max(0.0, min(turn.end, file_duration))
        if end_time_seg <= start_time_seg:
            continue
        segment = Segment(start_time_seg, end_time_seg)
        start_idx = int(start_time_seg * sample_rate)
        end_idx   = int(end_time_seg * sample_rate)
        segment_samples = audio_data[start_idx:end_idx]

        rms = np.sqrt(np.mean(segment_samples ** 2))
        if rms < 0.02:
            logger.debug(f"Segment energy too low ({rms:.4f}). Using 'Unknown' for speaker ID.")
            spk_id, similarity = "Unknown", 0.0
        else:
            with torch.no_grad():
                segment_embedding_torch = embedding_inference.crop(temp_wav, segment)
            segment_embedding = segment_embedding_torch.data
            spk_id, similarity = speaker_id_from_embedding(segment_embedding, speaker_db, threshold=0.3)

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

    final_text = "\n".join(f"{seg['speaker']} said: {seg['stt_text']}" for seg in segments_stats)
    total_stt_characters = sum(len(seg["stt_text"]) for seg in segments_stats)
    unique_speakers = list({seg["speaker"] for seg in segments_stats})
    
    # --- NEW: Aggregate similarity info per speaker (including 'Unknown') ---
    speaker_similarities: Dict[str, List[float]] = {}
    for seg in segments_stats:
        # Record similarity for all segments, even if speaker is 'Unknown'
        speaker_similarities.setdefault(seg["speaker"], []).append(seg["similarity"])
    # Compute average similarity per speaker.
    avg_speaker_similarities = {speaker: float(np.mean(sims)) for speaker, sims in speaker_similarities.items()}
    # -------------------------------------------------------------------------

    dia_stats = {
        "segments": segments_stats,
        "total_segments": len(segments_stats),
        "unique_speakers": unique_speakers,
        "total_stt_characters": total_stt_characters,
        "speaker_similarities": avg_speaker_similarities
    }
    total_elapsed = (time.perf_counter() - overall_start) * 1000
    logger.info(f"🔵 Diarization transcript:\n{final_text}\n(Total diarization+STT took {total_elapsed:.2f} ms)")
    return final_text, dia_stats

# =============================================================================
# Main Server
# =============================================================================

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
            model="openai/whisper-medium",
            device=self.device
        )

        self.llm_handler = LLamaConversationHandler(
            model_id="meta-llama/Meta-Llama-3-8B-Instruct"
        )

        self.tts_model, self.voice = build_kokoro_tts()
        # Adjust the path as needed.
        self.speaker_db = load_speaker_database("../datasets/speaker_embeddings.json")
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
                    total_loop_start = time.perf_counter()
                    audio_data = self._receive_audio()
                    if audio_data is None:
                        break  # Client disconnected

                    # Record audio received size (in bytes)
                    audio_received_bytes = audio_data.nbytes

                    # STT + Diarization with detailed stats
                    t_stt_start = time.perf_counter()
                    conversation_text, dia_stats = perform_diarization_stt(
                        audio_data,
                        sample_rate=SAMPLE_RATE,
                        stt_pipeline=self.stt_pipeline,
                        speaker_db=self.speaker_db
                    )
                    t_stt = (time.perf_counter() - t_stt_start) * 1000
                    logger.info(f"🔵 STT & Diarization completed in {t_stt:.2f} ms.")

                    # Process through LLM
                    t_llm_start = time.perf_counter()
                    logger.info("🟣 Interacting with LLM...")
                    response, probability = self.llm_handler.process_input(conversation_text)
                    t_llm = (time.perf_counter() - t_llm_start) * 1000
                    logger.info(f"🟣 LLM response (took {t_llm:.2f} ms):\n------------------\n{response}\n------------------")
                    # Count tokens in response using the tokenizer.
                    token_count = len(self.llm_handler.tokenizer.encode(response))

                    # Generate TTS audio from response (with phoneme info)
                    t_tts_start = time.perf_counter()
                    audio_response, tts_phonemes = generate_tts(self.tts_model, self.voice, response)
                    t_tts = (time.perf_counter() - t_tts_start) * 1000
                    logger.info(f"🟠 TTS generation completed in {t_tts:.2f} ms.")

                    # Determine next state based on LLM's probability
                    next_state = "WAKEWORD" if probability > 0.5 else "VAD"

                    # Compile conversation statistics into a dict
                    total_elapsed = (time.perf_counter() - total_loop_start) * 1000
                    conversation_stats = {
                        "timestamp": time.time(),
                        "audio_received_bytes": audio_received_bytes,
                        "stt_diarization_ms": t_stt,
                        "stt_characters": len(conversation_text),
                        "llm_ms": t_llm,
                        "llm_token_count": token_count,
                        "tts_ms": t_tts,
                        "tts_phonemes": tts_phonemes,
                        "total_processing_ms": total_elapsed,
                        "diarization": dia_stats
                    }
                    conversation_stats_list.append(conversation_stats)
                    logger.debug(f"Conversation stats: {json.dumps(conversation_stats, indent=2, cls=NumpyEncoder)}")

                    self._send_response(audio_response, next_state)
                    logger.info(f"Total processing time for this cycle: {total_elapsed:.2f} ms.\n")
        except KeyboardInterrupt:
            logger.info("Stopping server due to keyboard interrupt...")
        finally:
            if self.client_socket:
                self.client_socket.close()
            if self.server_socket:
                self.server_socket.close()
            logger.info("Server shut down.")
            # Dump the conversation_stats_list to a JSON file.
            with open("conversation_stats.json", "w") as f:
                json.dump(conversation_stats_list, f, indent=2, cls=NumpyEncoder)
            logger.info("Conversation statistics saved to 'conversation_stats.json'.")

if __name__ == "__main__":
    server = Server(SERVER_IP, SERVER_PORT)
    server.run()
