import pyaudio
import wave
import json
import numpy as np
import torch
from pyannote.audio import Model
from pyannote.core import Segment
from dotenv import load_dotenv
from pyannote.audio import Inference
import os

# Load environment variables
load_dotenv()

EMBEDDING_MODEL = "pyannote/embedding"
OUTPUT_JSON = "../datasets/speaker_embeddings.json"

HF_TOKEN = os.getenv("HF_AUTH_TOKEN", None)

# Number of enrollment samples to record per user
NUM_SAMPLES = 5
RECORD_SECONDS = 10
RATE = 16000
CHANNELS = 1
CHUNK = 1024
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def record_audio(filename, record_seconds=3):
    """Record audio from the default microphone and save to `filename`."""
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16,
                     channels=CHANNELS,
                     rate=RATE,
                     input=True,
                     frames_per_buffer=CHUNK)
    print(f"Recording {record_seconds} seconds to {filename}...")
    frames = []
    for _ in range(0, int(RATE / CHUNK * record_seconds)):
        data = stream.read(CHUNK)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    pa.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    print("Recording finished.")

def compute_embedding(wav_file, embedding_inference):
    """
    Compute the speaker embedding using the Inference crop method.
    This computes the embedding for the entire audio file.
    """
    wf = wave.open(wav_file, 'rb')
    n_frames = wf.getnframes()
    rate = wf.getframerate()
    duration = n_frames / rate
    wf.close()
    
    segment = Segment(0, duration)
    
    with torch.no_grad():
        embedding = embedding_inference.crop(wav_file, segment)
    
    # Return the embedding data (a numpy array)
    if hasattr(embedding, "data"):
        return embedding.data
    else:
        return embedding

if __name__ == "__main__":
    speaker_name = input("Enter the speaker's name you want to enroll (e.g., 'Mirko'): ")

    # Load the pyannote embedding model
    model = Model.from_pretrained(EMBEDDING_MODEL, use_auth_token=HF_TOKEN)
    embedding_inference = Inference(model, skip_aggregation=True, device=device)

    enrollment_embeddings = []
    for i in range(NUM_SAMPLES):
        fname = f"temp_{speaker_name}_{i}.wav"
        record_audio(fname, RECORD_SECONDS)
        emb = compute_embedding(fname, embedding_inference)
        # Average the embedding over frames if needed (to get a 1D vector)
        if emb.ndim > 1:
            emb = np.mean(emb, axis=0)
        enrollment_embeddings.append(emb)

    # Average across all enrollment samples to compute the final centroid
    centroid = np.mean(enrollment_embeddings, axis=0).tolist()

    # Load existing speaker database or initialize a new dict
    try:
        with open(OUTPUT_JSON, "r") as f:
            speaker_dict = json.load(f)
    except FileNotFoundError:
        speaker_dict = {}

    speaker_dict[speaker_name] = centroid

    with open(OUTPUT_JSON, "w") as f:
        json.dump(speaker_dict, f, indent=2)

    print(f"Enrolled {speaker_name} with centroid embedding. Saved to {OUTPUT_JSON}")
