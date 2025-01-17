import socket
import struct
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Tuple, List, Optional
from dataclasses import dataclass
import sys
import os
import json  # Import the json module
import base64

sys.path.append(os.path.abspath("../datasets/Kokoro-82M"))
from models import build_model
from kokoro import generate
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# --- Configuration ---
SERVER_IP = "127.0.0.1"  # Replace with the server's IP address if necessary
SERVER_PORT = 8080
SAMPLE_RATE = 24000  # Kokoro uses 24kHz audio
DEBUG = False
MODEL_PATH = "../datasets/Kokoro-82M/kokoro-v0_19.pth"  # Update with the correct path
VOICE_PATH = "../datasets/Kokoro-82M/voices/af.pt"
VOICE_NAME = "af"  # You can change this if using another voice

# --- LLM Agent ---
@dataclass
class ConversationState:
    context: List[dict]
    last_response: Optional[str] = None

class MistralConversationHandler:
    def __init__(self, model_path: str = "mistralai/Mistral-7B-Instruct-v0.1"):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, token=os.getenv("HF_AUTH_TOKEN")
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            token=os.getenv("HF_AUTH_TOKEN"),
        )
        self.conversation = ConversationState(context=[])

    def _build_prompt(self, user_input: str) -> str:
        history = ""
        for entry in self.conversation.context[-5:]:
            role = "User" if entry["role"] == "user" else "Assistant"
            history += f"{role}: {entry['content']}\n"

        prompt = f"""{history}
User: {user_input}

Consider if the user wants to end the conversation, and print it like this: PROBABILITY: [value]
Then, proceed with your response:"""
        return prompt

    def _parse_response(self, full_response: str) -> Tuple[str, float]:
        try:
            # Extract probability
            prob_start = full_response.find("PROBABILITY:")
            if prob_start == -1:
                return full_response, 0.0  # Return 0.0 if PROBABILITY is not found
            prob_end = full_response.find("\n", prob_start)
            prob_str = full_response[prob_start + len("PROBABILITY:"):prob_end].strip()
            probability = float(prob_str)

            # Extract response text
            response_text = full_response[prob_end + 1:].strip()
            # Remove also everything previous to "Your response:" to avoid leaking the prompt
            response_text = response_text[response_text.find("Your response:") + len("Your response:"):].strip()
            return response_text, probability
        except Exception as e:
            print(f"Error parsing response: {e}")
            return full_response, 0.0

    def generate_response(
        self, input_text: str, max_new_tokens: int = 512
    ) -> Tuple[str, float]:
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
        response_text, probability = self._parse_response(full_response)
        return response_text, probability

    def process_input(self, text: str) -> Tuple[str, bool]:
        self.conversation.context.append({"role": "user", "content": text})
        response, probability = self.generate_response(text)
        self.conversation.context.append({"role": "assistant", "content": response})
        should_end = probability > 0.5
        return response, should_end, probability

# --- Server Class ---
class Server:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.server_socket = None
        self.client_socket = None
        self.client_address = None
        self.stt_pipeline = pipeline(
            "automatic-speech-recognition", model="openai/whisper-tiny"
        )
        self.llm_handler = MistralConversationHandler()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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
        print(f"Receiving audio data of size: {data_size} bytes")

        # Receive the actual audio data
        audio_data = b""
        bytes_received = 0
        while bytes_received < data_size:
            chunk = self.client_socket.recv(
                min(4096, data_size - bytes_received)
            )  # Receive in chunks
            if not chunk:
                print("Connection closed by client during audio reception.")
                return None
            audio_data += chunk
            bytes_received += len(chunk)
        print(f"Received {bytes_received} bytes of audio data.")

        # Convert the received bytes to a NumPy array of float32
        audio_np = np.frombuffer(audio_data, dtype=np.float32)

        print(f"Audio data received successfully. Shape: {audio_np.shape}")

        return audio_np

    def _perform_stt(self, audio_data):
        print("Performing STT...")
        # Perform STT
        result = self.stt_pipeline(audio_data, chunk_length_s=30)
        text = result["text"]
        print(f"STT result: {text}")
        return text

    def _interact_with_llm(self, text):
        print("Interacting with LLM...")
        response, should_end, probability = self.llm_handler.process_input(text)
        print(f"LLM response: {response}")
        return response, should_end, probability

    def _generate_tts(self, text):
        print("Generating TTS audio with Kokoro...")
        try:
            audio, out_ps = generate(
                self.tts_model, text, self.voice, lang=VOICE_NAME[0]
            )
            print(f"Phonemes generated: {out_ps}")
            audio_np = audio
            audio_norm = audio_np.astype(np.float32) / np.iinfo(np.int16).max
            return audio_norm
        except Exception as e:
            print(f"Error in TTS generation: {e}")
            return None

    def _send_response(self, audio_data, next_state):
        if audio_data is not None:
            audio_base64 = base64.b64encode(audio_data.tobytes()).decode('utf-8')  # Encode to Base64
        else:
            audio_base64 = ""

        response_data = {
            "audio": audio_base64,
            "next_state": next_state
        }
        response_json = json.dumps(response_data)
        response_size = len(response_json)

        # Send the size of the JSON response
        self.client_socket.sendall(struct.pack("!I", response_size))

        # Send the JSON response
        self.client_socket.sendall(response_json.encode())

        print(f"Response sent with audio size {len(audio_base64)} bytes and next state '{next_state}'.")

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
                    response, should_end, probability = self._interact_with_llm(text)

                    # Determine the next state based on probability
                    if probability > 0.5:
                        next_state = "WAKEWORD"  # Go back to wake word detection
                    else:
                        next_state = "VAD"  # Continue with VAD

                    audio_response = self._generate_tts(response)
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