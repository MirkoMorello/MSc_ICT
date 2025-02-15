import pyaudio
import wave
import numpy as np
import simpleaudio as sa

def record_and_play(duration=5, sample_rate=16000, channels=1, chunk_size=1024, filename="test_recording.wav"):
    """Records audio from the microphone for a specified duration and plays it back."""

    p = pyaudio.PyAudio()

    stream = p.open(format=pyaudio.paInt16,  # Use Int16 for recording initially
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk_size)

    print("* recording")
    frames = []
    for _ in range(0, int(sample_rate / chunk_size * duration)):
        data = stream.read(chunk_size)
        frames.append(data)
    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save to a WAV file (for inspection)
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    # Play back the recorded audio
    wave_data = b''.join(frames)
    audio_np = np.frombuffer(wave_data, dtype=np.int16)
    play_obj = sa.play_buffer(audio_np, num_channels=channels, bytes_per_sample=2, sample_rate=sample_rate)
    play_obj.wait_done()

if __name__ == "__main__":
    record_and_play()