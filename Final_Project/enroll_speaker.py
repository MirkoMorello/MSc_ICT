import os
import json
import wave
import numpy as np
import pyaudio
import torch
from pyannote.audio import Model, Inference
from pyannote.core import Segment
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---

EMBEDDING_MODEL = "pyannote/embedding"
OUTPUT_JSON = "../datasets/speaker_embeddings.json"
HF_TOKEN = os.getenv("HF_AUTH_TOKEN", None)

RATE = 16000
CHANNELS = 1
CHUNK = 1024

NUM_ENROLL_SAMPLES = 20
ENROLL_RECORD_SECONDS = 3   # enrollment
IDENTIFY_RECORD_SECONDS = 5    # identification

IDENTIFY_WINDOW_DURATION = 2.0 # length of each sliding window
IDENTIFY_WINDOW_STEP = 0.5       # step between windows

# Similarity threshold for identification
IDENTIFY_THRESHOLD = 0.7

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# --- UTILITY FUNCTIONS ---

def record_audio(filename, record_seconds):
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
    Compute a speaker embedding for the entire audio file. we'll use this for enrollment, not identification.
    """
    # Determine the duration of the audio file
    wf = wave.open(wav_file, 'rb')
    n_frames = wf.getnframes()
    rate = wf.getframerate()
    duration = n_frames / rate
    wf.close()
    
    segment = Segment(0, duration)
    
    with torch.no_grad():
        embedding = embedding_inference.crop(wav_file, segment)
    
    # Extract underlying data from SlidingWindowFeature (if applicable)
    if hasattr(embedding, "data"):
        embedding = embedding.data
    
    # Convert to numpy array if it's a tensor
    if isinstance(embedding, torch.Tensor):
        embedding = embedding.cpu().numpy()
    
    # If multiple frames were returned, average them.
    if embedding.ndim > 1:
        print(f"[DEBUG] Enrollment embedding shape before aggregation: {embedding.shape}")
        embedding = np.mean(embedding, axis=0)
        print(f"[DEBUG] Enrollment embedding shape after aggregation: {embedding.shape}")
    
    # L2-normalize the embedding
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    else:
        print("[WARNING] Computed enrollment embedding has zero norm!")
    
    return embedding


def compute_embedding_sliding(wav_file, embedding_inference, window_duration, step):
    """
    Compute a robust speaker embedding from an audio file using a manual sliding window.
    For each window of duration `window_duration` (moving by step seconds), an embedding is computed using the model's crop() method. Then, all window embeddings are averaged and L2-normalized.
    """
    # Determine the duration of the audio file
    wf = wave.open(wav_file, 'rb')
    n_frames = wf.getnframes()
    rate = wf.getframerate()
    duration = n_frames / rate
    wf.close()
    
    embeddings = []
    start = 0.0
    while start + window_duration <= duration:
        segment = Segment(start, start + window_duration)
        with torch.no_grad():
            emb = embedding_inference.crop(wav_file, segment)
        if hasattr(emb, "data"):
            emb = emb.data
        if isinstance(emb, torch.Tensor):
            emb = emb.cpu().numpy()
        if emb.ndim > 1:
            # average embeddings over frames within the window.
            emb = np.mean(emb, axis=0)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        embeddings.append(emb)
        start += step

    if len(embeddings) > 0:
        aggregated = np.mean(np.array(embeddings), axis=0)
        norm = np.linalg.norm(aggregated)
        if norm > 0:
            aggregated = aggregated / norm
        return aggregated
    else:
        print("[WARNING] No embeddings extracted via sliding window!")
        return None


# --- CORE FUNCTIONS ---

def enroll_speaker(embedding_inference):
    speaker_name = input("Enter the speaker's name to enroll (e.g., 'Mirko'): ").strip()
    enrollment_embeddings = []

    for i in range(NUM_ENROLL_SAMPLES):
        fname = f"temp_{speaker_name}_{i}.wav"
        record_audio(fname, ENROLL_RECORD_SECONDS)
        emb = compute_embedding(fname, embedding_inference)
        print(f"[DEBUG] Enrollment sample {i+1}/{NUM_ENROLL_SAMPLES} embedding norm: {np.linalg.norm(emb):.4f}")
        enrollment_embeddings.append(emb)
        os.remove(fname) # clean up shite

    # compute the centroid of enrollment embeddings
    enrollment_embeddings = np.array(enrollment_embeddings)
    centroid = np.mean(enrollment_embeddings, axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm > 0:
        centroid = centroid / centroid_norm
    else:
        print("[WARNING] Centroid embedding has zero norm!")
    
    
    try:
        with open(OUTPUT_JSON, "r") as f:
            speaker_dict = json.load(f)
    except FileNotFoundError:
        speaker_dict = {}

    speaker_dict[speaker_name] = centroid.tolist()

    with open(OUTPUT_JSON, "w") as f:
        json.dump(speaker_dict, f, indent=2)

    print(f"Enrolled {speaker_name} with centroid embedding. Saved to {OUTPUT_JSON}")


def identify_speaker(embedding_inference, threshold=IDENTIFY_THRESHOLD):

    print("Identifying speaker. Please speak now...")
    fname = "temp_identify.wav"
    record_audio(fname, IDENTIFY_RECORD_SECONDS)
    
    # sliding window method for a robust embedding.
    emb = compute_embedding_sliding(fname, embedding_inference,
                                    window_duration=IDENTIFY_WINDOW_DURATION,
                                    step=IDENTIFY_WINDOW_STEP)
    os.remove(fname)
    
    try:
        with open(OUTPUT_JSON, "r") as f:
            speaker_dict = json.load(f)
    except FileNotFoundError:
        print("No enrolled speakers found. Please enroll first.")
        return None

    # compute cosine similarity (dot product, since embeddings are normalized)
    similarities = {}
    for speaker, centroid in speaker_dict.items():
        centroid = np.array(centroid)
        score = np.dot(emb, centroid)
        similarities[speaker] = score

    print("[DEBUG] Similarity scores:", similarities)

    # Determine best match
    best_speaker = max(similarities, key=similarities.get)
    best_score = similarities[best_speaker]
    if best_score < threshold:
        print("Speaker not identified confidently (score below threshold).")
        return None
    else:
        print("Identified speaker:", best_speaker, "with similarity:", best_score)
        return best_speaker


# --- MAIN ---

if __name__ == "__main__":
    mode = input("Select mode: (E)nroll or (I)dentify: ").strip().lower()

    print("Loading embedding model...")
    model = Model.from_pretrained(EMBEDDING_MODEL, use_auth_token=HF_TOKEN)
    embedding_inference = Inference(model, skip_aggregation=True, device=device)
    print("Model loaded.")

    if mode.startswith('e'):
        enroll_speaker(embedding_inference)
    elif mode.startswith('i'):
        identify_speaker(embedding_inference)
    else:
        print("Invalid mode selected. Please choose 'E' for enrollment or 'I' for identification.")
