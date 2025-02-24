import pyaudio
import wave
import numpy as np
import simpleaudio as sa

def record_and_save_amplified(duration=5, sample_rate=16000, channels=1, chunk_size=1024,
                              amplified_filename="amplified_recording.wav",
                              amplification_factor=2.0, normalize=False):
    """
    Records audio from the microphone, amplifies it (with optional normalization),
    saves the amplified audio to a WAV file, and plays back the processed audio.
    """

    # Initialize PyAudio and open the stream for recording
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk_size)

    print("* Recording...")
    frames = []
    num_chunks = int(sample_rate / chunk_size * duration)
    for _ in range(num_chunks):
        data = stream.read(chunk_size, exception_on_overflow=False)
        frames.append(data)
    print("* Done recording.")

    stream.stop_stream()
    stream.close()
    p.terminate()


    wave_data = b''.join(frames)

    audio_np = np.frombuffer(wave_data, dtype=np.int16).astype(np.float32)


    audio_np *= amplification_factor
    print(f"Applied fixed amplification factor: {amplification_factor}")

    if normalize:
        max_val = np.max(np.abs(audio_np))
        if max_val > 0:
            normalization_factor = 32767 / max_val
            audio_np *= normalization_factor
            print(f"Applied dynamic normalization factor: {normalization_factor:.2f}")
        else:
            print("Warning: Max amplitude is 0; skipping normalization.")


    audio_np = np.clip(audio_np, -32768, 32767).astype(np.int16)


    amplified_bytes = audio_np.tobytes()


    with wave.open(amplified_filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(amplified_bytes)
    print(f"Amplified recording saved as '{amplified_filename}'.")

    print("* Playing amplified audio...")
    play_obj = sa.play_buffer(audio_np, num_channels=channels, bytes_per_sample=2, sample_rate=sample_rate)
    play_obj.wait_done()
    print("* Playback finished.")

if __name__ == "__main__":
    record_and_save_amplified(amplification_factor=50.0, normalize=True)
