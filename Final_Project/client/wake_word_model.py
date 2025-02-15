from EncDecClassificationModel import EncDecClassificationModel
import torch

def get_model(): 
    model = EncDecClassificationModel(num_classes=35, sample_rate=16000)
    state_dict = torch.load("best_model.pth", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model