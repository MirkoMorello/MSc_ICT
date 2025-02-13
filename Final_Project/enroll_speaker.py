import pyaudio
import wave
import json
import numpy as np
import torch
from pyannote.audio import Model

EMBEDDING_MODEL = "pyannote/embedding"
OUTPUT_JSON = "./datasets/speaker_embeddings.json"

# Number of enrollment samples to record per user
NUM_SAMPLES = 3
RECORD_SECONDS = 3
RATE = 24000
CHANNELS = 1
CHUNK = 1024

def record_audio(filename, record_seconds=3):
    """Record audio from default microphone and save to `filename`."""
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

def compute_embedding(wav_file, embedding_model):
    """Compute speaker embedding of a WAV file with PyAnnote embedding model."""
    # Load audio as torch tensor
    # Each model can differ, but typically you do something like:
    audio_data, sample_rate = embedding_model.audio(wav_file)

    #TODO: Check if we need to trim the audio data
    # We'll assume the entire file is the segment for enrollment, do we have to trim?
    embedding = embedding_model(audio_data, sample_rate)
    return embedding

def main():
    speaker_name = input("Enter the speaker's name you want to enroll (e.g., 'Mirko'): ")

    model = Model.from_pretrained(EMBEDDING_MODEL)
    embeddings = []

    for i in range(NUM_SAMPLES):
        fname = f"temp_{speaker_name}_{i}.wav"
        record_audio(fname, RECORD_SECONDS)
        emb = compute_embedding(fname, model)
        embeddings.append(emb.detach().cpu().numpy())

    # Average (centroid) embedding
    centroid = np.mean(embeddings, axis=0).tolist()

    # Load or initialize JSON
    try:
        with open(OUTPUT_JSON, "r") as f:
            speaker_dict = json.load(f)
    except FileNotFoundError:
        speaker_dict = {}

    # Store new centroid
    speaker_dict[speaker_name] = centroid

    with open(OUTPUT_JSON, "w") as f:
        json.dump(speaker_dict, f, indent=2)

    print(f"Enrolled {speaker_name} with centroid embedding. Saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
