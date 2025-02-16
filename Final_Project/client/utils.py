from models import EncDecClassificationModel
import numpy as np
import torch

def get_model(path): 
    model = EncDecClassificationModel(num_classes=35, sample_rate=16000)
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model

def audio_amplifier(audio_chunk):
    audio_np = audio_chunk * 50
    audio_np = np.clip(audio_np, -32768, 32767).astype(np.int16)
    return audio_np