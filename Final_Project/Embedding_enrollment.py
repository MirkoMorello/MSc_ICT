import torch
import torchaudio
from pyannote.audio import Pipeline
from dotenv import load_dotenv
import json
import argparse
import numpy as np
import os

load_dotenv()
SAMPLE_RATE = 16000

def get_embedding_model(auth_token):
    """Loads the embedding model from pyannote."""
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.0", use_auth_token=auth_token)
    return pipeline.get_embedding_model()

def extract_embedding(model, audio_file):
    """Extracts the speaker embedding from an audio file."""
    try:
        waveform, sample_rate = torchaudio.load(audio_file)
        if sample_rate != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
            waveform = resampler(waveform)

        # Check if the waveform is stereo and convert to mono if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        waveform = waveform.unsqueeze(0)  # Add a batch dimension
        with torch.no_grad():
            embedding = model(waveform).squeeze().cpu().numpy()
        return embedding
    except Exception as e:
        print(f"Error extracting embedding from {audio_file}: {e}")
        return None

def main(audio_files, name, auth_token):
    model = get_embedding_model(auth_token)
    embeddings = [extract_embedding(model, audio_file) for audio_file in audio_files]
    embeddings = [e for e in embeddings if e is not None]  # Filter out None values
    if not embeddings:
        print("No valid embeddings found.")
        return

    # Calculate the average embedding
    avg_embedding = np.mean(embeddings, axis=0)

    # Enroll the speaker with the averaged embedding
    try:
        with open("enrolled_embeddings.json", "r") as f:
            enrolled_embeddings = json.load(f)
    except FileNotFoundError:
        enrolled_embeddings = {}

    enrolled_embeddings[name] = avg_embedding.tolist()

    with open("enrolled_embeddings.json", "w") as f:
        json.dump(enrolled_embeddings, f)

    print(f"Enrolled speaker '{name}' with {len(embeddings)} audio files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enroll a speaker by extracting embeddings from audio files.")
    parser.add_argument("audio_files", nargs="+", help="List of audio files to enroll from.")
    parser.add_argument("--name", required=True, help="Name of the speaker to enroll.")
    parser.add_argument("--auth_token", help="Hugging Face authentication token.")
    args = parser.parse_args()
    
    auth_token = args.auth_token or os.getenv("HF_AUTH_TOKEN") # lazy evaluation, we're safe
    if not auth_token:
        raise ValueError("Authentication token must be provided either via --auth_token argument or HF_AUTH_TOKEN environment variable")


    main(args.audio_files, args.name, args.auth_token)