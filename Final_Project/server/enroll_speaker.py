# enroll.py
import sys
import os
import logging
import wave
import numpy as np
import pyaudio
import torch
from pyannote.audio import Model as EmbeddingModel
from pyannote.audio import Inference

# Import our common utilities
from utils import (
    load_speaker_database,
    save_speaker_database,
    extract_sliding_embedding_for_segment
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
JSON_DB_PATH = "speaker_embeddings.json"

def record_audio(filename, record_seconds=3):
    """Record from the microphone and save to a WAV file."""
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16,
                     channels=1,
                     rate=SAMPLE_RATE,
                     input=True,
                     frames_per_buffer=1024)
    logger.info(f"Recording {record_seconds} seconds to '{filename}'...")
    frames = []
    for _ in range(int(SAMPLE_RATE / 1024 * record_seconds)):
        data = stream.read(1024)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    pa.terminate()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(frames))
    logger.info(f"Recorded audio saved to '{filename}'.")

def enroll_speaker(
    speaker_name: str,
    embedding_inference: Inference,
    db_path: str = JSON_DB_PATH,
    num_samples=3,
    record_seconds=3
):
    """
    Enroll a new speaker by recording 'num_samples' short utterances,
    extracting embeddings with the same sliding approach,
    and averaging them to form a single centroid.
    """
    spk_db = load_speaker_database(db_path)
    all_vectors = []

    for i in range(num_samples):
        temp_wav = f"enroll_{speaker_name}_{i}.wav"
        record_audio(temp_wav, record_seconds)

        # figure out total duration
        with wave.open(temp_wav, 'rb') as wf:
            frames = wf.getnframes()
            sr = wf.getframerate()
            duration = frames / sr

        # Use sliding-window approach for the entire utterance
        seg_emb = extract_sliding_embedding_for_segment(
            temp_wav,
            0.0,
            duration,
            embedding_inference
        )
        logger.info(f"[Sample {i+1}/{num_samples}] norm={np.linalg.norm(seg_emb):.3f}")
        all_vectors.append(seg_emb)

        os.remove(temp_wav)

    # average across all enrollment samples
    centroid = np.mean(np.array(all_vectors), axis=0)
    c_norm = np.linalg.norm(centroid)
    if c_norm > 0:
        centroid /= c_norm

    # store in DB
    spk_db[speaker_name] = centroid.tolist()
    save_speaker_database(spk_db, db_path)
    logger.info(f"Speaker '{speaker_name}' enrolled successfully with {num_samples} samples.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python enroll.py [speaker_name]")
        sys.exit(1)

    speaker_name = sys.argv[1].strip()
    logger.info(f"Enrolling speaker '{speaker_name}'...")

    # Create the model and inference object for sliding-window embeddings

    HF_TOKEN = os.getenv("HF_AUTH_TOKEN", None)
    try:
        embedding_model = EmbeddingModel.from_pretrained(
            "pyannote/embedding",
            token=HF_TOKEN
        )
        logger.info("Embedding model loaded successfully.")
    except Exception as e:
        logger.warning(f"Could not load pyannote/embedding: {e}")
        sys.exit(1)

    embedding_inference = Inference(
        embedding_model,
        skip_aggregation=True,    # we handle averaging ourselves
        window="sliding",
        duration=1.5,
        step=0.75,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Now do the enrollment
    enroll_speaker(
        speaker_name=speaker_name,
        embedding_inference=embedding_inference,
        db_path=JSON_DB_PATH,
        num_samples=3,
        record_seconds=3
    )
