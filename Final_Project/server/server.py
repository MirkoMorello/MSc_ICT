import socket
import struct
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Tuple, List, Optional
from dataclasses import dataclass
import sys
import os
sys.path.append(os.path.abspath('../datasets/Kokoro-82M'))
from models import build_model
from kokoro import generate
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file


# --- Configuration ---
SERVER_IP = "127.0.0.1"  # Replace with the server's IP address if necessary
SERVER_PORT = 12345
SAMPLE_RATE = 24000  # Kokoro uses 24kHz audio
DEBUG = False
MODEL_PATH = "../datasets/Kokoro-82M/kokoro-v0_19.pth"  # Update with the correct path
VOICE_PATH = "../datasets/Kokoro-82M/voices/af.pt"
VOICE_NAME = 'af' # You can change this if using another voice
# --- LLM Agent ---
@dataclass
class ConversationState:
    context: List[dict]
    last_response: Optional[str] = None

class MistralConversationHandler:
    def __init__(self, model_path: str = "mistralai/Mistral-7B-Instruct-v0.1"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                       use_auth_token=os.getenv("HF_AUTH_TOKEN"))
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            use_auth_token=os.getenv("HF_AUTH_TOKEN")
        )
        self.conversation = ConversationState(context=[])

    def _build_prompt(self, user_input: str, user_info: str = "None") -> str:
        history = ""
        for entry in self.conversation.context[-5:]:
            role = "User" if entry["role"] == "user" else "Assistant"
            history += f"{role}: {entry['content']}\n"
        prompt = f"""Previous conversation:
        {history}
        User Information:
        {user_info}
        User: {user_input}
        Think step by step:
        1. Is this conversation likely to end naturally? Consider:
        - Did the user express intention to leave (e.g., "That's all," "Goodbye")?
        - Is this a natural conclusion point (e.g., question answered, task completed)?
        - Has the main topic been resolved?
        2. Should I end this conversation based on the user's engagement and social context? Analyze:
        - User's tone and sentiment
        - Completeness of the interaction
        - Typical conversation norms
        Decision Confidence (0.0 - 1.0): [Provide a confidence score for ending the conversation]
        Assistant's response:"""
        return prompt

    def _parse_response(self, full_response: str) -> Tuple[str, bool]:
        try:
            parts = full_response.split("Assistant's response:")
            decision_part = parts[0].lower()
            should_end = False
            if "decision confidence:" in decision_part:
                confidence_score = float(
                    decision_part.split("decision confidence:")[1].split("]")[0].strip()
                )
                should_end = confidence_score >= 0.7
            response = parts[1].strip() if len(parts) > 1 else "I understand."
            return response, should_end
        except Exception as e:
            print(f"Error parsing response: {e}")
            return "I understand.", False

    def generate_response(
        self, input_text: str, user_info: str = "None", max_new_tokens: int = 512
    ) -> Tuple[str, bool]:
        prompt = self._build_prompt(input_text, user_info)
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
        response_text = full_response[
            len(self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)) :
        ]
        response, should_end = self._parse_response(response_text)
        return response, should_end

    def process_input(self, text: str, user_info: str = "None") -> Tuple[str, bool]:
        self.conversation.context.append({"role": "user", "content": text})
        response, should_end = self.generate_response(text, user_info)
        self.conversation.context.append({"role": "assistant", "content": response})
        if should_end and not any(
            word in response.lower() for word in ["goodbye", "bye", "farewell"]
        ):
            response += "\nGoodbye! Have a great day!"
        return response, should_end

# --- Server Class ---
class Server:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.server_socket = None
        self.client_socket = None
        self.client_address = None
        self.stt_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
        self.llm_handler = MistralConversationHandler()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tts_model = build_model(MODEL_PATH, self.device)
        self.voice = torch.load(VOICE_PATH, weights_only=True).to(self.device)

    def _start_server(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.ip, self.port))
        self.server_socket.listen(1)
        print(f"Server started on {self.ip}:{self.port}. Waiting for connection...")

    def _accept_client(self):
        self.client_socket, self.client_address = self.server_socket.accept()
        print(f"Connection from {self.client_address} established.")

    def _receive_audio(self):
        # Receive the size of the audio data first (an integer)
        data_size_bytes = self.client_socket.recv(4)
        if not data_size_bytes:
            print("Connection closed by client.")
            return None
        data_size = struct.unpack("!I", data_size_bytes)[0]

        # Receive the actual audio data
        audio_data = b""
        bytes_received = 0
        while bytes_received < data_size:
            chunk = self.client_socket.recv(min(4096, data_size - bytes_received)) # Receive in chunks
            if not chunk:
                print("Connection closed by client during audio reception.")
                return None
            audio_data += chunk
            bytes_received += len(chunk)

        # Convert the received bytes to a NumPy array of int16
        audio_np = np.frombuffer(audio_data, dtype=np.int16)

        # Reshape the audio data to 2D array with a single channel for mono audio
        audio_np = audio_np.reshape((1, -1))

        return audio_np

    def _perform_stt(self, audio_data):
        print("Performing STT...")
        # Convert audio to float32 and normalize
        audio_data_float32 = audio_data.astype(np.float32) / 32768.0
        # Perform STT
        result = self.stt_pipeline(audio_data_float32[0], chunk_length_s=30)
        text = result["text"]
        print(f"STT result: {text}")
        return text

    def _interact_with_llm(self, text):
        print("Interacting with LLM...")
        response, should_end = self.llm_handler.process_input(text)
        print(f"LLM response: {response}")
        return response, should_end
    
    def _generate_tts(self, text):
        print("Generating TTS audio with Kokoro...")
        try:
            audio, out_ps = generate(self.tts_model, text, self.voice, lang=VOICE_NAME[0])
            audio_np = audio.cpu().numpy()
            # Convert to float32 with range -1.0 to 1.0
            audio_norm = audio_np.astype(np.float32) / np.iinfo(np.int16).max
            
            return audio_norm
        except Exception as e:
            print(f"Error in TTS generation: {e}")
            return None

    def _send_audio(self, audio_data):
        # Convert audio data to bytes
        audio_bytes = audio_data.tobytes()

        # Send the size of the audio data first
        data_size = len(audio_bytes)
        self.client_socket.sendall(struct.pack("!I", data_size))

        # Send the actual audio data
        self.client_socket.sendall(audio_bytes)

    def run(self):
        self._start_server()
        try:
            while True:
                self._accept_client()
                while True:
                    audio_data = self._receive_audio()
                    if audio_data is None:
                        break  # Client disconnected

                    text = self._perform_stt(audio_data)
                    response, should_end = self._interact_with_llm(text)
                    audio_response = self._generate_tts(response)
                    self._send_audio(audio_response)
                    if should_end:
                        break
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