import os
import argparse
import torch
import torchio as tio
import torch.nn as nn
import nibabel as nib
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd

from model import SingleModel

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on a given NIfTI image.")
    parser.add_argument("image_path", type=str, help="Path to the NIfTI image file.")
    parser.add_argument("mode", type=str, choices=["avg", "var"], help="Mode of inference: 'avg' for average, 'var' for variance-based segmentation.")
    return parser.parse_args()

def save_heatmap(image, save_path):
    fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=100) 
    ax.imshow(image.T, cmap='jet', interpolation='nearest', vmin = 0, vmax = 1)
    ax.axis('off')  # Remove axes for cleaner images
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)

def run_inference(model, image_path, mode, device): 
    ### set up the cropping function
    cropping = tio.CropOrPad((320,320,32), padding_mode=0)

    ### create a folder to store outputs
    os.makedirs("results", exist_ok=True)
    
    ### Load and process image
    nii_img = nib.load(image_path)
    affine = nii_img.affine
    img_data = nii_img.get_fdata()
    shape = img_data.shape
    img_tensor = torch.tensor(img_data).unsqueeze(0)
    img_tensor = cropping(img_tensor).float().squeeze().numpy()
    img_tensor = torch.tensor(img_tensor).to(device)

    with torch.no_grad():
        ### find the output and mean output
        outputs = model(img_tensor.unsqueeze(0).unsqueeze(0))
        mean_voxel = outputs.mean(dim=1).squeeze().cpu().detach().numpy()
        if mode == "avg":
            mean_voxel = outputs.mean(dim=1).squeeze().cpu().detach().numpy()
            seg = (mean_voxel > threshold).astype(np.float32)
        elif mode == "var":
            ### find the standard deviations
            std_voxel = outputs.std(dim=1).squeeze().cpu().detach().numpy()    # shape: (1, H, W, D)

            ### normalize the standard deviation to make variance relative to the image
            std_voxel = (std_voxel - np.min(std_voxel))/(np.max(std_voxel) - np.min(std_voxel) + 1e-8)

            ### remove high variance regions from the mean voxel
            seg = (mean_voxel > 0.5).astype(np.float32) - (std_voxel > 0.75).astype(np.float32)
            seg[seg < 0] = 0

            ### save each slice as a jpg
            for d in range(outputs.shape[4]):
                mean_slice = mean_voxel[ :, :, d] 
                std_slice = std_voxel[ :, :, d]
                os.makedirs(os.path.join("results", "heatmaps_mean"), exist_ok=True)
                os.makedirs(os.path.join("results", "heatmaps_std"), exist_ok=True)
                save_heatmap(mean_slice, os.path.join("results", "heatmaps_mean", f"mean_{d}.jpg"))
                save_heatmap(std_slice, os.path.join("results", "heatmaps_std", f"std_{d}.jpg"))

    else:
        raise Exception("mode not valid; either avg or var")

    ### save the segmentation
    nii_seg = nib.Nifti1Image(seg, affine)
    nib.save(nii_seg, os.path.join("results", f"predicted.nii.gz")) 

if __name__ == "__main__":
    args = parse_args()

    ### verify torch and cuda
    print(torch.__version__)
    print("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.version.cuda)
    print(torch.backends.cudnn.enabled)

    ### load model
    model = SingleModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("weights/model_weights.pth", weights_only=True))
    model.to(device)

    ### run inference
    run_inference(model, args.image_path, args.mode, device)
