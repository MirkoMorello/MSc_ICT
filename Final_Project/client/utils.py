from models import EncDecClassificationModel
import torch

def get_model(path): 
    model = EncDecClassificationModel(num_classes=35, sample_rate=16000)
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model