import torch
from model import SingleModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
normal_model = SingleModel().to(device)
checkpoint = torch.load("weights/base_3d_unet.pt", map_location=device)
new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
normal_model.load_state_dict(new_state_dict)
torch.save(normal_model.state_dict(), "weights/model_weights.pth")