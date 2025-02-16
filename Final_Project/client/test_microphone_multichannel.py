import pigpio
import numpy as np
import scipy.signal as sps
import wave
import time

# Initialize pigpio
pi = pigpio.pi()

# Pin assignments
MIC_CK = 18  # Clock signal (shared)
MIC_D0 = 21  # Mic 1 & 2 (Stereo)
MIC_D1 = 20  # Mic 3 & 4 (Stereo)
MIC_D2 = 26  # Mic 5 & 6 (Stereo)
MIC_D3 = 16  # Reference Mic (Mono)

# Sampling parameters
SAMPLE_RATE = 16000  # 16 kHz
DURATION = 5  # Record for 5 seconds
BUFFER = []
NUM_SAMPLES = SAMPLE_RATE * DURATION  # Total samples to collect

# Beamforming delay values (adjust based on mic geometry)
MIC_POSITIONS = np.array([[-0.05, 0], [0, 0], [0.05, 0]])  # 3-mic pairs along X-axis
SOUND_SPEED = 343  # Speed of sound in m/s

# Callback function for reading microphones
def sample_mics(gpio, level, tick):
    if len(BUFFER) < NUM_SAMPLES:
        gpio_levels = pi.read_bank_1()
        
        # Extract stereo microphone pairs
        d0 = (gpio_levels >> MIC_D0) & 1
        d1 = (gpio_levels >> MIC_D1) & 1
        d2 = (gpio_levels >> MIC_D2) & 1
        d3 = (gpio_levels >> MIC_D3) & 1  # Reference mic
        
        # Store samples in a buffer
        BUFFER.append([d0, d1, d2, d3])

# Set GPIO modes
pi.set_mode(MIC_CK, pigpio.INPUT)
pi.set_mode(MIC_D0, pigpio.INPUT)
pi.set_mode(MIC_D1, pigpio.INPUT)
pi.set_mode(MIC_D2, pigpio.INPUT)
pi.set_mode(MIC_D3, pigpio.INPUT)

# Attach interrupt to MIC_CK (trigger on rising edge)
pi.callback(MIC_CK, pigpio.RISING_EDGE, sample_mics)

# Record for DURATION seconds
time.sleep(DURATION)

# Stop pigpio
pi.stop()

# Convert BUFFER to numpy array
raw_data = np.array(BUFFER, dtype=np.int16)

# Separate microphone channels
mic1, mic2 = raw_data[:, 0], raw_data[:, 1]
mic3, mic4 = raw_data[:, 2], raw_data[:, 3]

# Interleave stereo pairs
mic1, mic2 = mic1[::2], mic1[1::2]  # Left and right from MIC_D0
mic3, mic4 = mic3[::2], mic3[1::2]  # Left and right from MIC_D1
mic5, mic6 = mic4[::2], mic4[1::2]  # Left and right from MIC_D2
ref_mic = mic3  # Reference mic (mono)

# -------------------------
# 🌟 Beamforming (Delay-and-Sum)
# -------------------------
def delay_and_sum(mics, sample_rate, mic_positions, sound_speed):
    num_mics = len(mics)
    center_mic = mic_positions[num_mics // 2]  # Assume center mic
    beamformed = np.zeros_like(mics[0], dtype=np.float32)

    for i, mic in enumerate(mics):
        distance = np.linalg.norm(mic_positions[i] - center_mic)
        time_delay = distance / sound_speed  # Delay in seconds
        sample_delay = int(time_delay * sample_rate)  # Convert to sample delay
        delayed_signal = np.roll(mic, sample_delay)
        beamformed += delayed_signal

    return beamformed / num_mics  # Normalize

# Apply beamforming
beamformed_audio = delay_and_sum(
    [mic1, mic2, mic3, mic4, mic5, mic6], SAMPLE_RATE, MIC_POSITIONS, SOUND_SPEED
)

# -------------------------
# 🌟 Noise Reduction (Spectral Subtraction)
# -------------------------
def spectral_subtraction(signal, noise, alpha=1.5):
    signal_fft = np.fft.rfft(signal)
    noise_fft = np.fft.rfft(noise)

    clean_fft = signal_fft - alpha * noise_fft
    clean_fft = np.maximum(clean_fft, 0)  # Ensure non-negative values

    return np.fft.irfft(clean_fft).astype(np.int16)

# Apply spectral subtraction using ref mic as noise reference
denoised_audio = spectral_subtraction(beamformed_audio, ref_mic)

# -------------------------
# 🌟 Save as WAV file
# -------------------------
def save_wav(filename, data, sample_rate):
    with wave.open(filename, "w") as wf:
        wf.setnchannels(1)  # Mono output
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())

# Save the processed audio
save_wav("output.wav", denoised_audio, SAMPLE_RATE)

print("Recording complete. Processed audio saved as output.wav 🎤🎧")
