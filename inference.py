### load the torch and torchvision libraries
import os
import torch
import torchio as tio
import torch.nn as nn


from loss import MultiLoss
from model import SingleModel

import nibabel as nib
from tqdm import tqdm

import os


print(torch.__version__)
print("cuda" if torch.cuda.is_available() else "cpu")
print(torch.version.cuda)  # Check CUDA version
print(torch.backends.cudnn.enabled)

### from the class to the object
model = SingleModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print("let's use", torch.cuda.device_count(), "GPUs")
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load("weights/base_3d_unet.pt", weights_only=True))
model.to(device)


def run_inference(model, source_path): #scheduler
    cropping = tio.CropOrPad((320,320,32), padding_mode=0)
    for img_path in tqdm(os.listdir(source_path)):
        base_path = os.path.join("generated", img_path[:-7])
        os.makedirs(base_path, exist_ok=False)

        nii_img = nib.load(os.path.join(source_path, img_path))
        affine = nii_img.affine
        img_data = nii_img.get_fdata()
        img_tensor = torch.tensor(img_data).unsqueeze(0)
        img_tensor = cropping(img_tensor).float().unsqueeze(0).to(device)
        with torch.no_grad():
            #tensor = torch.tensor(img_tensor, dtype=torch.float32)
            outputs = model(img_tensor)
            for channel in range(outputs.shape[1]):
                seg = outputs[0][channel]
                seg = (seg > 0.5).float().clone().detach().cpu().numpy()
                nii_seg = nib.Nifti1Image(seg, affine)
                nib.save(nii_seg, os.path.join(base_path, f"{channel + 1}.nii.gz"))


run_inference(model=model, source_path="Cirrhosis_T2_3D/test_images/") 
