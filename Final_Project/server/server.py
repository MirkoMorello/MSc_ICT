import socket
import struct
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Tuple, List, Optional
from dataclasses import dataclass
import sys
import os
import json
import base64
import wave
from pyannote.core import Segment

from pyannote.audio import Pipeline
from pyannote.audio import Model as EmbeddingModel
from pyannote.audio import Inference
from pyannote.core import Segment


sys.path.append(os.path.abspath("../datasets/Kokoro-82M"))
from models import build_model
from kokoro import generate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SERVER_IP = "127.0.0.1"
SERVER_PORT = 8080
SAMPLE_RATE = 24000
DEBUG = False
MODEL_PATH = "../datasets/Kokoro-82M/kokoro-v0_19.pth"
VOICE_PATH = "../datasets/Kokoro-82M/voices/af.pt"
VOICE_NAME = "af"
HF_TOKEN = os.getenv("HF_AUTH_TOKEN", None)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

try:
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN
    )
except Exception as e:
    print(f"Could not load pyannote/speaker-diarization-3.1: {e}")
    diarization_pipeline = None

try:
    embedding_model = EmbeddingModel.from_pretrained(
        "pyannote/embedding",
        use_auth_token=HF_TOKEN
    )
except Exception as e:
    print(f"Could not load pyannote/embedding: {e}")
    embedding_model = None
    

if embedding_model is not None:
    embedding_inference = Inference(
        embedding_model,
        skip_aggregation=True, 
        device=device
    )
else:
    embedding_inference = None



@dataclass
class ConversationState:
    context: List[dict]
    last_response: Optional[str] = None

class MistralConversationHandler:
    def __init__(self, model_path: str = "mistralai/Mistral-7B-Instruct-v0.1"):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            token=os.getenv("HF_AUTH_TOKEN")
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            token=os.getenv("HF_AUTH_TOKEN"),
        )
        self.conversation = ConversationState(context=[])

    def _build_prompt(self, user_input: str) -> str:
        system_prompt = (
            "You are a helpful and friendly AI assistant. "
            "Engage in a natural conversation with the user, "
            "providing informative and relevant responses. "
            "The conversation history is provided below."
            "Try to infer if the user wants to end the conversation "
            "and only answer if you are sure."
            "If present, the user name will be prepended to the input with 'user said: '"
        )
        conversation_history = ""
        for entry in self.conversation.context[-5:]:
            if entry["role"] == "user":
                conversation_history += f"User: {entry['content']}\n"
            else:
                conversation_history += f"Assistant: {entry['content']}\n"

        prompt = f"""<s>[INST] {system_prompt}

Conversation History:
{conversation_history}

Current User Input:
{user_input} [/INST]
"""
        return prompt

    def _parse_response(self, full_response: str, user_input: str) -> Tuple[str, float]:
        response_text = full_response.split("[/INST]")[-1].strip()
        if response_text.startswith("Assistant:"):
            response_text = response_text[len("Assistant:"):].strip()

        end_conversation_keywords = ["bye", "goodbye", "stop", "end"]
        probability = 0.0
        if any(keyword in user_input.lower() for keyword in end_conversation_keywords):
            probability = 0.9
        return response_text, probability

    def generate_response(self, input_text: str, max_new_tokens: int = 512) -> Tuple[str, float]:
        prompt = self._build_prompt(input_text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text, probability = self._parse_response(full_response, input_text)
        return response_text, probability

    def process_input(self, text: str) -> Tuple[str, float]:
        self.conversation.context.append({"role": "user", "content": text})
        response, probability = self.generate_response(text)
        self.conversation.context.append({"role": "assistant", "content": response})
        return response, probability

def build_kokoro_tts():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts_model = build_model(MODEL_PATH, device)
    voice_obj = torch.load(VOICE_PATH, weights_only=True).to(device)
    return tts_model, voice_obj

def generate_tts(tts_model, voice, text):
    print("Generating TTS audio with Kokoro...")
    try:
        audio, out_ps = generate(tts_model, text, voice, lang=VOICE_NAME[0])
        print(f"Phonemes generated: {out_ps}")
        audio_np = audio
        audio_norm = audio_np.astype(np.float32) / np.iinfo(np.int16).max
        return audio_norm
    except Exception as e:
        print(f"Error in TTS generation: {e}")
        return None

def load_speaker_database(json_path="speaker_embeddings.json"):
    if not os.path.isfile(json_path):
        print(f"Speaker database not found at {json_path}.")
        return {}
    with open(json_path, "r") as f:
        print(f"Loading speaker database from {json_path}")
        return json.load(f)

def speaker_id_from_embedding(embedding_vector: np.ndarray, speaker_db: dict, threshold=0.7) -> str:
    if not speaker_db:
        return "Unknown"
    
    # If the embedding is multi-frame (e.g. shape (T, 512)), aggregate by averaging along the time axis.
    if embedding_vector.ndim > 1:
        embedding_vector = np.mean(embedding_vector, axis=0)
    
    # Ensure we have a 1D vector and normalize it.
    emb = embedding_vector.squeeze()
    emb_norm = emb / (np.linalg.norm(emb) + 1e-10)
    print(f"[DEBUG] Enrollment embedding norm: {np.linalg.norm(emb):.4f}")
    
    best_spk = "Unknown"
    best_score = -1.0

    print("Similarity scores for each enrolled speaker:")
    for spk_name, centroid in speaker_db.items():
        centroid = np.array(centroid)
        # If the stored centroid is multi-frame, average it as well.
        if centroid.ndim > 1:
            centroid = np.mean(centroid, axis=0)
        centroid = centroid.squeeze()
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-10)
        
        # Compute cosine similarity
        sim = np.dot(emb_norm, centroid_norm)
        print(f"  {spk_name}: {sim:.4f} (centroid norm: {np.linalg.norm(centroid):.4f})")
        
        if sim > best_score:
            best_score = sim
            best_spk = spk_name
            
    print(f"[DEBUG] Best similarity: {best_score:.4f}")
    if best_score < threshold:
        print(f"Speaker similarity below threshold: {best_score:.4f}")
        return "Unknown"
    
    return best_spk



# Suppose you do this globally, once:
# embedding_inference = Inference(embedding_model, skip_aggregation=True, device=device)

def perform_diarization_stt(
    audio_data: np.ndarray,
    sample_rate: int,
    stt_pipeline,
    speaker_db: dict
) -> str:
    """
    If diarization_pipeline or embedding_inference is None, fallback to entire STT.
    Otherwise, for each diarized speaker turn, check its energy. If the segment’s RMS
    energy is below a threshold (indicating silence or very low speech), mark its speaker
    as "Unknown" without computing an embedding. Otherwise, compute the embedding,
    perform speaker identification, and run STT on the segment.
    """
    
    if diarization_pipeline is None or embedding_inference is None:
        print("WARNING: Diarization or embedding model not loaded. Using entire STT only.")
        result = stt_pipeline(
            {"raw": audio_data, "sampling_rate": sample_rate},
            chunk_length_s=30
        )
        return result["text"]

    # 1) Save audio to a temporary WAV file
    temp_wav = "temp_received.wav"
    with wave.open(temp_wav, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        int16_data = (audio_data * 32767).astype(np.int16)
        wf.writeframes(int16_data.tobytes())

    # 2) Perform diarization on the saved WAV file
    diarization = diarization_pipeline(temp_wav)
    speaker_transcripts = []

    # Set an energy threshold (tune this value as needed)
    energy_threshold = 0.02  # Example threshold: adjust based on your audio scale

    # 3) Process each speaker turn
    for turn, _, speaker_label in diarization.itertracks(yield_label=True):
        start_time = turn.start
        end_time = turn.end
        segment = Segment(start_time, end_time)

        # 4) Determine the indices for this segment in the audio data
        start_idx = int(start_time * sample_rate)
        end_idx = int(end_time * sample_rate)
        segment_samples = audio_data[start_idx:end_idx]

        # Compute RMS energy of the segment
        rms = np.sqrt(np.mean(segment_samples**2))
        if rms < energy_threshold:
            print(f"Segment energy too low ({rms:.4f}). Using 'Unknown' for speaker ID.")
            spk_id = "Unknown"
        else:
            with torch.no_grad():
                segment_embedding_torch = embedding_inference.crop(temp_wav, segment)
            segment_embedding = segment_embedding_torch.data
            spk_id = speaker_id_from_embedding(segment_embedding, speaker_db, threshold=0.2)

        # 5) Run STT on just that segment
        stt_result = stt_pipeline(
            {"raw": segment_samples, "sampling_rate": sample_rate},
            chunk_length_s=30
        )
        text = stt_result["text"]

        speaker_transcripts.append(f"{spk_id} said: {text}")

    final_text = "\n".join(speaker_transcripts)
    print("Diarization transcript:\n", final_text)
    return final_text



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

        self.llm_handler = MistralConversationHandler()
        self.tts_model, self.voice = build_kokoro_tts()

        self.speaker_db = load_speaker_database("../datasets/speaker_embeddings.json")

    def _start_server(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.ip, self.port))
        self.server_socket.listen(1)
        print(f"Server started on {self.ip}:{self.port}. Waiting for connection...")

    def _accept_client(self):
        self.client_socket, self.client_address = self.server_socket.accept()
        print(f"Connection from {self.client_address} established.")

    def _receive_audio(self):
        data_size_bytes = self.client_socket.recv(4)
        if not data_size_bytes:
            print("Connection closed by client.")
            return None
        data_size = struct.unpack("!I", data_size_bytes)[0]
        print(f"Receiving audio data of size: {data_size} bytes")

        audio_data = b""
        bytes_received = 0
        while bytes_received < data_size:
            chunk = self.client_socket.recv(min(4096, data_size - bytes_received))
            if not chunk:
                print("Connection closed by client during audio reception.")
                return None
            audio_data += chunk
            bytes_received += len(chunk)
        print(f"Received {bytes_received} bytes of audio data.")

        audio_np = np.frombuffer(audio_data, dtype=np.float32)
        print(f"Audio data received successfully. Shape: {audio_np.shape}")
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

        if DEBUG:
            print(f"Sending response_size: {response_size} bytes")

        self.client_socket.sendall(struct.pack("!I", response_size))
        self.client_socket.sendall(response_json.encode())
        print(f"Response sent with audio size {len(audio_base64)} bytes and next state '{next_state}'.\n\n")

    def run(self):
        self._start_server()
        try:
            while True:
                self._accept_client()
                while True:
                    audio_data = self._receive_audio()
                    if audio_data is None:
                        break  # Client disconnected

                    conversation_text = perform_diarization_stt(
                        audio_data,
                        sample_rate=SAMPLE_RATE,
                        stt_pipeline=self.stt_pipeline,
                        speaker_db=self.speaker_db
                    )

                    print("Interacting with LLM...")
                    response, probability = self.llm_handler.process_input(conversation_text)
                    print(f"LLM response\n------------------\n{response}\n------------------")

                    audio_response = generate_tts(self.tts_model, self.voice, response)

                    if probability > 0.5:
                        next_state = "WAKEWORD"
                    else:
                        next_state = "VAD"

                    self._send_response(audio_response, next_state)
        except KeyboardInterrupt:
            print("Stopping server...")
        finally:
            if self.client_socket:
                self.client_socket.close()
            if self.server_socket:
                self.server_socket.close()

if __name__ == "__main__":
    server = Server(SERVER_IP, SERVER_PORT)
    server.run()
