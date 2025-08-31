# Marvin: A Full-Stack AI Virtual Assistant

> **Course:** Intelligent Consumer Technologies

## Overview

Marvin is a complete, real-time, voice-controlled AI assistant, built from the ground up. The project encompasses hardware design, embedded machine learning, and a sophisticated, scalable server-side AI pipeline. The system is designed around a client-server architecture where a low-power client captures audio and handles wake-word detection, while a powerful server performs the heavy lifting of natural language understanding and response generation.

## System Architecture

### Client (The Device)

The client is a custom-built, self-contained smart speaker.

*   **Hardware:**
    *   Raspberry Pi 3 Model B+
    *   7-microphone array for audio acquisition
    *   Stereo speakers and amplifier for audio output
    *   Lithium battery with a power management module
    *   Custom-designed, 3D-printed enclosure
*   **On-Device AI (Wake-Word Detection):**
    *   To ensure privacy and reduce server load, wake-word detection runs entirely on the Raspberry Pi.
    *   We designed a highly efficient, **MatchboxNet-inspired CNN with only 77k parameters**. This custom model achieves 94% accuracy on the Google Speech Commands V2 dataset while being small and fast enough for real-time inference on the resource-constrained Raspberry Pi.

### Server (The Brain)

The server is designed to handle multiple clients concurrently with minimal latency, orchestrating a state-of-the-art conversational AI pipeline over a custom, low-overhead binary TCP protocol.

*   **Server Pipeline:**
    1.  **Speaker Diarization:** Segments incoming audio to identify *who* is speaking and when, using `pyannote.audio`.
    2.  **Speech-to-Text (STT):** Transcribes each speaker's segments into text using `Whisper Large v3 Turbo`.
    3.  **LLM Processing:** Aggregates the multi-speaker conversation context and feeds it to a quantized `Llama 8B` model to generate a coherent, context-aware response.
    4.  **Text-to-Speech (TTS):** Converts the LLM's text response back into natural-sounding audio using `Kokoro TTS`.

## Technologies Used

*   **Hardware:** Raspberry Pi 3, Custom PCBs, 3D Printing
*   **On-Device AI:** PyTorch, Custom CNNs for Keyword Spotting
*   **Server AI:** Python, `pyannote.audio`, OpenAI `whisper`, `transformers` (Llama 8B), `Kokoro TTS`
*   **Networking:** Custom TCP binary protocol
