# server/audio_processing.py
import os
import wave
import numpy as np
import torch
import logging
from typing import Tuple, Dict, Any, List
from pyannote.audio import Pipeline
from pyannote.audio import Model as EmbeddingModel
from pyannote.audio import Inference
from transformers import pipeline
# Corrected: Import ONLY the specific names you need.
from .utils.speaker_id import load_speaker_database, extract_sliding_embedding_for_segment, speaker_id_from_embedding
from .utils.audio_utils import save_audio_to_wav
from .utils import logging_utils  # Keep this one, you use the module directly for get_logger
from .config import HF_TOKEN, SAMPLE_RATE, SPEAKER_EMBEDDINGS_FILE

logger = logging_utils.get_logger(__name__)

try:
    from pyannote.audio.pipelines.utils.resegmentation import Resegmentation
except ImportError:
    Resegmentation = None

class AudioProcessor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stt_pipeline = None
        self.diarization_pipeline = None
        self.embedding_model = None
        self.embedding_inference = None
        self.resegmenter = None
        self.speaker_db = {}
        self.load_models()
        self.load_speaker_db()

    def load_models(self):
        try:
            self.stt_pipeline = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-large-v3-turbo",
                device=self.device
            )
            logger.info("STT pipeline loaded successfully.")
        except Exception as e:
            logger.error(f"Could not load STT pipeline: {e}")

        try:
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=HF_TOKEN
            )
            logger.info("Diarization pipeline loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load pyannote/speaker-diarization-3.1: {e}")

        try:
            self.embedding_model = EmbeddingModel.from_pretrained(
                "pyannote/embedding",
                token=HF_TOKEN
            )
            if self.embedding_model is not None:
              self.embedding_inference = Inference(
                    self.embedding_model,
                    skip_aggregation=True,
                    device=self.device,
                    window="sliding",
                    duration=1.5,
                    step=0.75
                )
            logger.info("Embedding model loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load pyannote/embedding: {e}")

        if Resegmentation and self.embedding_model:
            self.resegmenter = Resegmentation(segmentation=self.embedding_model)
            logger.info("Resegmentation module instantiated.")
        else:
            logger.warning("Resegmentation module not available.")

    def load_speaker_db(self):
        self.speaker_db = load_speaker_database(SPEAKER_EMBEDDINGS_FILE)

    def perform_diarization_stt(self, audio_data: np.ndarray, sample_rate: int) -> Tuple[str, Dict[str, Any]]:
        if self.stt_pipeline is None:
            logger.error("STT pipeline is not loaded. Cannot process audio.")
            return "STT unavailable", {}

        segments_stats: List[Dict[str, Any]] = []
        temp_wav = "temp_received.wav"

        try:
            save_audio_to_wav(audio_data, sample_rate, temp_wav)
        except Exception as e:
            logger.error(f"Failed to save temporary WAV file: {e}")
            return "Audio processing error", {}

        file_duration = len(audio_data) / sample_rate

        if self.diarization_pipeline is None or self.embedding_inference is None:
            logger.warning("Diarization or embedding not loaded; using entire-file STT only.")
            try:
                result = self.stt_pipeline({"raw": audio_data, "sampling_rate": sample_rate}, chunk_length_s=30)
                dia_stats = {
                    "segments": [{
                        "start": 0.0,
                        "end": file_duration,
                        "speaker": "Fallback",
                        "similarity": None,
                        "stt_text": result["text"],
                        "rms": np.sqrt(np.mean(audio_data ** 2)),
                    }],
                    "total_segments": 1,
                    "unique_speakers": ["Fallback"],
                    "total_stt_characters": len(result["text"])
                }
                return result["text"], dia_stats
            except Exception as e:
                logger.error(f"Error during fallback STT: {e}")
                return "STT error", {}

        try:
            diarization = self.diarization_pipeline(temp_wav)
            diarization = self.resegment_with_embeddings(diarization, temp_wav)
            raw_segments = []
            for turn, _, label in diarization.itertracks(yield_label=True):
                start_seg = max(0.0, min(turn.start, file_duration))
                end_seg = max(0.0, min(turn.end, file_duration))
                if end_seg > start_seg:
                    raw_segments.append((start_seg, end_seg, label))
            raw_segments.sort(key=lambda x: x[0])

            for (start_time_seg, end_time_seg, diar_label) in raw_segments:
                start_idx = int(start_time_seg * sample_rate)
                end_idx = int(end_time_seg * sample_rate)
                segment_samples = audio_data[start_idx:end_idx]
                rms = np.sqrt(np.mean(segment_samples ** 2))

                seg_emb = extract_sliding_embedding_for_segment(
                    temp_wav, start_time_seg, end_time_seg, self.embedding_inference
                )

                if seg_emb.size > 0:
                    spk_id, similarity = speaker_id_from_embedding(seg_emb, self.speaker_db, threshold=0.25)
                else:
                    spk_id, similarity = "Unknown", 0.0

                stt_result = self.stt_pipeline({"raw": segment_samples, "sampling_rate": sample_rate}, chunk_length_s=30)
                text = stt_result["text"]

                segments_stats.append({
                    "start": start_time_seg,
                    "end": end_time_seg,
                    "speaker": spk_id,
                    "similarity": similarity,
                    "stt_text": text,
                    "rms": rms,
                })
            merged_segments = self.merge_short_segments(segments_stats)
            final_text = "\n".join(f"{seg['speaker']} said: {seg['stt_text']}" for seg in merged_segments)
            total_stt_characters = sum(len(seg["stt_text"]) for seg in merged_segments)
            unique_speakers = list({seg["speaker"] for seg in merged_segments})

            speaker_similarities: Dict[str, List[float]] = {}
            for seg in merged_segments:
                speaker_similarities.setdefault(seg["speaker"], []).append(seg["similarity"])
            avg_speaker_similarities = {
                spk: float(np.mean(sims)) for spk, sims in speaker_similarities.items() if sims[0] is not None
            }

            dia_stats = {
                "segments": merged_segments,
                "total_segments": len(merged_segments),
                "unique_speakers": unique_speakers,
                "total_stt_characters": total_stt_characters,
                "speaker_similarities": avg_speaker_similarities
            }
            return final_text, dia_stats

        except Exception as e:
            logger.exception(f"Error during diarization/STT/speaker ID: {e}")
            return "Diarization/STT error", {}

    def resegment_with_embeddings(self, diar_result, wav_path):
        if self.resegmenter is None:
            logger.debug("No resegmentation available. Skipping.")
            return diar_result
        logger.debug("Performing embedding-based re-segmentation...")
        try:
            return self.resegmenter(wav_path, diar_result)
        except Exception as e:
          logger.error(f"Resegmentation failed: {e}")
          return diar_result

    def merge_short_segments(self, segments, min_duration_merge=0.7):
        if not segments:
            return []
        merged_segments = []
        prev = segments[0]
        for current in segments[1:]:
            duration = current["end"] - current["start"]
            if current["speaker"] == prev["speaker"] and duration < min_duration_merge:
                prev["end"] = current["end"]
                prev["stt_text"] = prev["stt_text"].strip() + " " + current["stt_text"].strip()
                if prev["similarity"] is not None and current["similarity"] is not None:
                    prev["similarity"] = max(prev["similarity"], current["similarity"])
                elif current["similarity"] is not None:
                    prev["similarity"] = current["similarity"]
            else:
                merged_segments.append(prev)
                prev = current
        merged_segments.append(prev)
        return merged_segments