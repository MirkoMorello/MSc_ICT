import torch
import wave
import numpy as np
from Final_Project.client.utils import get_model  # Your model loading function
from torchinfo import summary

def test_model_with_wav(wav_filepath, model_path="path/to/your/traced_model.pt", labels=None):
    """
    Reads a WAV file, processes the audio, and feeds it to the wake word model.
    """

    # --- 1. Load the Model ---
    model = get_model(path=model_path)  # Load your model
    print(summary(model, input_data=(torch.randn(1, 16000) , torch.tensor([16000]))))  # Display model summary
    if model is None:
        print("Error: Model loading failed.")
        return None
    model.eval() 

    # --- 2. Load the WAV file ---
    try:
        with wave.open(wav_filepath, 'rb') as wf:
            num_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frame_rate = wf.getframerate()
            num_frames = wf.getnframes()
            raw_data = wf.readframes(num_frames)

            
            if sample_width != 2:
                print(f"Error: Unsupported sample width ({sample_width}).  Expected 2 (int16).")
                return None

            audio_data = np.frombuffer(raw_data, dtype=np.int16)

            # handle multi-channel audio
            if num_channels > 1:
                print("Warning: Multi-channel audio. Using only the first channel.")
                audio_data = audio_data[::num_channels] # Take every nth sample

    except wave.Error as e:
        print(f"Error reading WAV file: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

    # --- 3. Preprocess the Audio (Convert to float32 and normalize) ---
    audio_data = audio_data.astype(np.float32) / 32768.0 # Convert to float32 and rescale

    # --- 4. Prepare for the Model (unsqueeze to create a batch) ---
    audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)  # Add a batch dimension
    audio_length = torch.tensor([audio_tensor.size(1)]) #Length must be a tensor.

    # --- 5. Run Inference ---
    try:
        with torch.no_grad():
            logits = model(audio_tensor, audio_length)
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class_index = torch.argmax(probabilities, dim=-1).item()

            if labels:  
                predicted_label = labels[predicted_class_index]
                print(f"Predicted label: {predicted_label}")
                return predicted_label
            else:
                print(f"Predicted class index: {predicted_class_index}")
                return predicted_class_index 

    except Exception as e:
        print(f"Error during model inference: {e}")
        return None

if __name__ == "__main__":
    labels = [
    'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
    'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin',
    'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six',
    'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'
    ] 
    test_model_with_wav("test_recording.wav", labels=labels, model_path = "../best_model.pth") 
    # test_model_with_wav("vad_test_recording.wav", model_path="your_model.pt", labels=your_labels)