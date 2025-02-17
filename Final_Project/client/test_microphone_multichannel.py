import pigpio
import numpy as np
import wave
import time
import scipy.signal as sps
import sys

# ---------------------------
# Configuration & Pin Setup
# ---------------------------
# Pin assignments:
MIC_CK = 18   # I2S clock
MIC_WS = 19   # I2S word-select (determines left/right)
MIC_D0 = 21   # Data for Mic Pair 1 (2 channels)
MIC_D1 = 20   # Data for Mic Pair 2 (2 channels)
MIC_D2 = 26   # Data for Mic Pair 3 (2 channels)
MIC_D3 = 16   # Data for Reference Mic (mono)

# Sampling parameters:
SAMPLE_RATE = 16000   # desired output sample rate in Hz
DURATION = 5          # seconds to record
NUM_SAMPLES = SAMPLE_RATE * DURATION  # target number of audio samples
BITS_PER_SAMPLE = 16  # we decode 16 bits per sample

# ---------------------------
# Global Buffers & Temporary Storage
# ---------------------------
# We will form 6 beamforming channels (from 3 stereo pairs) plus 1 reference.
buffer_beam = [[] for _ in range(6)]  # channels: D0_left, D0_right, D1_left, D1_right, D2_left, D2_right
buffer_ref = []                      # reference channel (from D3)

# Temporary bit accumulators for each channel:
temp_bits = {
    'D0_left': [],
    'D0_right': [],
    'D1_left': [],
    'D1_right': [],
    'D2_left': [],
    'D2_right': [],
    'D3': []  # Reference mic (mono; no WS used)
}

# ---------------------------
# I²S Bit-Decoding Callback
# ---------------------------
def bits_to_int(bits):
    """Convert a list of bits (MSB first) to a signed integer."""
    val = 0
    for b in bits:
        val = (val << 1) | b
    if val & (1 << (BITS_PER_SAMPLE - 1)):
        val -= (1 << BITS_PER_SAMPLE)
    return val

def i2s_callback(gpio, level, tick):
    global temp_bits, buffer_beam, buffer_ref
    # Read word select to know which channel (0: left, 1: right)
    ws = pi.read(MIC_WS)
    # Read each data pin’s current bit:
    bit_D0 = pi.read(MIC_D0)
    bit_D1 = pi.read(MIC_D1)
    bit_D2 = pi.read(MIC_D2)
    bit_D3 = pi.read(MIC_D3)  # Reference

    # For stereo pins, accumulate bits based on WS:
    if ws == 0:
        temp_bits['D0_left'].append(bit_D0)
        temp_bits['D1_left'].append(bit_D1)
        temp_bits['D2_left'].append(bit_D2)
    else:
        temp_bits['D0_right'].append(bit_D0)
        temp_bits['D1_right'].append(bit_D1)
        temp_bits['D2_right'].append(bit_D2)
    # For the reference mic, simply accumulate bits:
    temp_bits['D3'].append(bit_D3)

    # Check if we have collected a full word (16 bits) for the left side.
    if len(temp_bits['D0_left']) >= BITS_PER_SAMPLE:
        # Decode left-channel samples from each stereo data pin:
        sample_D0_left = bits_to_int(temp_bits['D0_left'][:BITS_PER_SAMPLE])
        sample_D1_left = bits_to_int(temp_bits['D1_left'][:BITS_PER_SAMPLE])
        sample_D2_left = bits_to_int(temp_bits['D2_left'][:BITS_PER_SAMPLE])
        # Remove the used bits:
        temp_bits['D0_left'] = temp_bits['D0_left'][BITS_PER_SAMPLE:]
        temp_bits['D1_left'] = temp_bits['D1_left'][BITS_PER_SAMPLE:]
        temp_bits['D2_left'] = temp_bits['D2_left'][BITS_PER_SAMPLE:]
        # For the reference mic:
        if len(temp_bits['D3']) >= BITS_PER_SAMPLE:
            sample_ref = bits_to_int(temp_bits['D3'][:BITS_PER_SAMPLE])
            temp_bits['D3'] = temp_bits['D3'][BITS_PER_SAMPLE:]
        else:
            sample_ref = 0

        # Similarly, if the right-channel buffers have a full word:
        if len(temp_bits['D0_right']) >= BITS_PER_SAMPLE:
            sample_D0_right = bits_to_int(temp_bits['D0_right'][:BITS_PER_SAMPLE])
            sample_D1_right = bits_to_int(temp_bits['D1_right'][:BITS_PER_SAMPLE])
            sample_D2_right = bits_to_int(temp_bits['D2_right'][:BITS_PER_SAMPLE])
            temp_bits['D0_right'] = temp_bits['D0_right'][BITS_PER_SAMPLE:]
            temp_bits['D1_right'] = temp_bits['D1_right'][BITS_PER_SAMPLE:]
            temp_bits['D2_right'] = temp_bits['D2_right'][BITS_PER_SAMPLE:]
        else:
            sample_D0_right = 0
            sample_D1_right = 0
            sample_D2_right = 0

        # Append the decoded 16-bit samples to the appropriate beamforming channels.
        # Order: 0: D0_left, 1: D0_right, 2: D1_left, 3: D1_right, 4: D2_left, 5: D2_right.
        buffer_beam[0].append(sample_D0_left)
        buffer_beam[1].append(sample_D0_right)
        buffer_beam[2].append(sample_D1_left)
        buffer_beam[3].append(sample_D1_right)
        buffer_beam[4].append(sample_D2_left)
        buffer_beam[5].append(sample_D2_right)
        # Append the reference sample:
        buffer_ref.append(sample_ref)

# ---------------------------
# Initialize pigpio and Set Pin Modes
# ---------------------------
pi = pigpio.pi()
if not pi.connected:
    raise Exception("Could not connect to pigpio daemon!")

for pin in [MIC_CK, MIC_WS, MIC_D0, MIC_D1, MIC_D2, MIC_D3]:
    pi.set_mode(pin, pigpio.INPUT)

# Attach the callback on MIC_CK’s rising edge:
cb = pi.callback(MIC_CK, pigpio.RISING_EDGE, i2s_callback)

# ---------------------------
# Record for the Desired Duration
# ---------------------------
print("Recording for {} seconds...".format(DURATION))
time.sleep(DURATION)

# Cancel the callback and stop pigpio
cb.cancel()
pi.stop()

# ---------------------------
# Convert Collected Data to Numpy Arrays & Trim to Equal Length
# ---------------------------
# Check that we collected some data
if not buffer_beam[0]:
    print("No data was collected. Check your wiring or timing.")
    sys.exit(1)

# For beamforming channels:
min_len = min(len(chan) for chan in buffer_beam)
if min_len == 0:
    print("Error: One or more beamforming channels have zero length.")
    sys.exit(1)

beam_channels = [np.array(chan[:min_len], dtype=np.int16) for chan in buffer_beam]

# For reference channel:
ref_channel = np.array(buffer_ref, dtype=np.int16)
ref_channel = ref_channel[:min_len]

# Debug: print lengths
print("Beamforming channels length:", [len(m) for m in beam_channels])
print("Reference channel length:", len(ref_channel))

# ---------------------------
# Beamforming: Circular Array Delay-and-Sum
# ---------------------------
# For a circular array of 6 microphones:
RADIUS = 0.05         # 5 cm radius
NUM_BEAM = 6          # 6 beamforming channels
SOUND_SPEED = 343.0   # m/s
# Compute evenly spaced angles for the 6 mics:
ANGLES = np.linspace(0, 2*np.pi, NUM_BEAM, endpoint=False)
TARGET_DIRECTION = 0  # steer toward 0 radians (front)
# Compute time delays in seconds, then convert to sample delays:
delays_sec = (RADIUS * np.cos(TARGET_DIRECTION - ANGLES)) / SOUND_SPEED
sample_delays = (delays_sec * SAMPLE_RATE).astype(int)

def circular_beamforming(mics, sample_rate, sample_delays):
    num_mics = len(mics)
    min_length = min(len(mic) for mic in mics)
    # Trim all channels to the same length:
    mics = [mic[:min_length] for mic in mics]
    beamformed = np.zeros(min_length, dtype=np.float32)
    for i, mic in enumerate(mics):
        # Apply delay (using np.roll) and trim to min_length:
        delayed_signal = np.roll(mic, sample_delays[i])[:min_length]
        beamformed += delayed_signal
    return beamformed / num_mics

beamformed_audio = circular_beamforming(beam_channels, SAMPLE_RATE, sample_delays)

# Ensure beamformed_audio is nonempty
if beamformed_audio.size == 0:
    print("Error: Beamformed audio has zero length.")
    sys.exit(1)

# ---------------------------
# Noise Reduction: Spectral Subtraction
# ---------------------------
def spectral_subtraction(signal, noise, alpha=1.5):
    if signal.size == 0:
        raise ValueError("Signal is empty. Cannot perform FFT.")
    # Compute FFTs:
    signal_fft = np.fft.rfft(signal)
    noise_fft = np.fft.rfft(noise)
    # Subtract a scaled noise spectrum:
    clean_fft = signal_fft - alpha * noise_fft
    # Ensure non-negative magnitude:
    clean_fft = np.maximum(clean_fft, 0)
    # Reconstruct time-domain signal:
    clean_signal = np.fft.irfft(clean_fft)
    return clean_signal.astype(np.int16)

try:
    denoised_audio = spectral_subtraction(beamformed_audio, ref_channel)
except ValueError as e:
    print("Error during spectral subtraction:", e)
    sys.exit(1)

# ---------------------------
# Save Processed Audio as a WAV File
# ---------------------------
def save_wav(filename, data, sample_rate):
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)      # Mono output
        wf.setsampwidth(2)      # 16-bit PCM
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())

output_filename = "circular_output.wav"
save_wav(output_filename, denoised_audio, SAMPLE_RATE)
print("Recording complete. Processed audio saved as '{}'".format(output_filename))
