# utils.py
import os
import json
import logging
import numpy as np
import torch
from pyannote.audio import Model as EmbeddingModel
from pyannote.audio import Inference
from pyannote.core import Segment

logger = logging.getLogger(__name__)

def load_speaker_database(json_path="speaker_embeddings.json"):
    """Load speaker database from disk or return empty dict if file not found."""
    if not os.path.isfile(json_path):
        logger.warning(f"Speaker database not found at {json_path}. Returning empty DB.")
        return {}
    with open(json_path, "r") as f:
        logger.info(f"Loading speaker database from {json_path}")
        return json.load(f)

def save_speaker_database(spk_db, json_path="speaker_embeddings.json"):
    """Save speaker database (dict) to disk as JSON."""
    with open(json_path, "w") as f:
        json.dump(spk_db, f, indent=2)
    logger.info(f"Speaker database saved to {json_path}")

def extract_sliding_embedding_for_segment(
    wav_path: str,
    start_time: float,
    end_time: float,
    inference_engine: Inference
) -> np.ndarray:
    """
    Use pyannote's sliding-window approach (skip_aggregation=True)
    to extract multiple frames from [start_time, end_time], then average them.
    Finally, L2-normalize. Returns a 1D np.array of shape (D,).
    """
    with torch.no_grad():
        seg_embedding = inference_engine.crop(wav_path, Segment(start_time, end_time))

    # If it's a SlidingWindowFeature, get .data
    if hasattr(seg_embedding, "data"):
        frames = seg_embedding.data  # shape (N, D)
    else:
        frames = seg_embedding  # could be a tensor or np.array

    if isinstance(frames, torch.Tensor):
        frames = frames.cpu().numpy()  # shape (N, D) or (D,)

    if frames.ndim == 1:
        # only one frame => shape (D,)
        emb = frames
    else:
        # multiple frames => average over time => shape (D,)
        emb = frames.mean(axis=0)

    # L2-normalize
    norm_val = np.linalg.norm(emb)
    if norm_val > 0:
        emb = emb / norm_val
    else:
        logger.warning("Zero-norm embedding encountered in extract_sliding_embedding_for_segment.")
    return emb

def speaker_id_from_embedding(
    embedding_vector: np.ndarray,
    speaker_db: dict,
    threshold=0.25
):
    if not speaker_db:
        return "Unknown", 0.0

    # Normalize the input embedding
    emb_norm = np.linalg.norm(embedding_vector)
    if emb_norm > 0:
        emb_vector = embedding_vector / emb_norm
    else:
        emb_vector = embedding_vector

    best_spk = "Unknown"
    best_score = -1.0

    for spk_name, centroids in speaker_db.items():
        for centroid in centroids:
            centroid_arr = np.array(centroid, dtype=np.float32)
            c_norm = np.linalg.norm(centroid_arr)
            if c_norm > 0:
                centroid_arr /= c_norm

            sim = float(np.dot(emb_vector, centroid_arr))
            if sim > best_score:
                best_score = sim
                best_spk = spk_name

    if best_score < threshold:
        return "Unknown", best_score
    return best_spk, best_score
