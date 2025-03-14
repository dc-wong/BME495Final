import os
import torch
import torchio as tio
import torch.nn as nn
import nibabel as nib
from tqdm import tqdm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

import pandas as pd

from model import SingleModel
from metrics import JaccardIndex, ASSD, Dice, Precision, Recall, HD95

print(torch.__version__)
print("cuda" if torch.cuda.is_available() else "cpu")
print(torch.version.cuda)  # Check CUDA version
print(torch.backends.cudnn.enabled)

# Instantiate model and load weights
model = SingleModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print("let's use", torch.cuda.device_count(), "GPUs")
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load("weights/base_3d_unet.pt", weights_only=True))
model.to(device)





def save_heatmap(image, save_path):
    """Save a 2D heatmap as a JPG image."""
    fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=100) 
    ax.imshow(image, cmap='jet', interpolation='nearest')
    ax.axis('off')  # Remove axes for cleaner images
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)

def run_inference(model, image_path, label_path, threshold, mode, results):
    # Define cropping/padding transformation to standard size
    cropping = tio.CropOrPad((320, 320, 32), padding_mode=0)
    
    for img_filename in tqdm(os.listdir(image_path)):
        base_path = os.path.join("results/generated_" + str(int(threshold * 100)), img_filename[:-7])
        os.makedirs(base_path, exist_ok=True)
        
        # Load and process image
        nii_img = nib.load(os.path.join(image_path, img_filename))
        affine = nii_img.affine
        img_data = nii_img.get_fdata()
        img_tensor = torch.tensor(img_data).unsqueeze(0)
        img_tensor = cropping(img_tensor).float().unsqueeze(0).to(device)
        nii_img_cropped = nib.Nifti1Image(img_tensor, affine)
        nib.save(nii_img_cropped, os.path.join(base_path, "mri.nii.gz"))
        
        with torch.no_grad():
            outputs = model(img_tensor)  # outputs shape: (1, n_channels, H, W, D)
            
            if mode == "stat":
                n_channels = outputs.shape[1]
                # Compute mean and std across channels for each voxel
                mean_voxel = outputs.mean(dim=1)  # shape: (1, H, W, D)
                std_voxel = outputs.std(dim=1)    # shape: (1, H, W, D)
                # Avoid division by zero by adding a small epsilon
                epsilon = 1e-8
                # Compute t-statistic: (mean - 1) / (std / sqrt(n_channels))
                t_stat = (mean_voxel - 0.5) / (std_voxel / (n_channels ** 0.5) + epsilon)
                # Convert t_stat to numpy array for SciPy
                t_stat = t_stat.cpu().numpy().squeeze()  # shape: (H, W, D)
                
                # Compute one-sided p-value (probability that mean > 1)
                # degrees of freedom = n_channels - 1
                p_value = 1 - stats.t.cdf(t_stat, df=n_channels - 1)

                print(p_value.min(), p_value.mean(), p_value.std(), p_value.max())

                for d in range(outputs.shape[3]):
                    mean_slice = mean_voxel[ :, :, d]  # Shape: (H, W)
                    std_slice = std_voxel[ :, :, d]    # Shape: (H, W)
                    pval_slice = p_value[ :, :, d]     # Shape: (H, W)

                    save_heatmap(mean_slice, os.path.join(base_path, "heatmaps_mean", f"mean_{d}.jpg"))
                    save_heatmap(std_slice, os.path.join(base_path, "heatmaps_std", f"std_{d}.jpg"))
                    save_heatmap(pval_slice, os.path.join(base_path, "heatmaps_pval", f"pval_{d}.jpg"))
            elif mode == "avg":
                p_value = outputs.mean(dim=1)
            else:
                raise Exception("mode is either avg or stat")
            
            # Create segmentation mask: 1 if p-value is below the threshold, else 0
            seg = (p_value < threshold).astype(np.float32)
            
            # Save the segmentation mask as a NIfTI file
            nii_seg = nib.Nifti1Image(seg, affine)
            nib.save(nii_seg, os.path.join(base_path, f"prediction_{mode}_{str(int(threshold * 100))}.nii.gz"))
        
        # Process the label similarly (only cropping here)
        nii_label = nib.load(os.path.join(label_path, img_filename))
        affine_label = nii_label.affine
        label_data = nii_label.get_fdata()
        label_tensor = torch.tensor(label_data).unsqueeze(0)
        label_data_cropped = cropping(label_tensor).squeeze().numpy()
        nii_label_cropped = nib.Nifti1Image(label_data_cropped, affine_label)
        nib.save(nii_label_cropped, os.path.join(base_path, "mask.nii.gz"))

        preds = torch.tensor(seg).unsqueeze(0).unsqueeze(0).to(device)
        target = torch.tensor(label_data_cropped).unsqueeze(0).unsqueeze(0).to(device)
        
        jaccard = JaccardIndex(preds, target).item()
        dice = Dice(preds, target).item()
        precision = Precision(preds, target).item()
        recall = Recall(preds, target).item()
        assd = ASSD(preds, target).item()
        hd95 = HD95(preds, target).item()
        
        row_name = f"{img_filename} | mode: {mode}, threshold: {threshold}"
        
        results[row_name] = {
            "Jaccard Index": jaccard,
            "Dice": dice,
            "Precision": precision,
            "Recall": recall,
            "ASSD": assd,
            "HD95": hd95
        }

results = {}
# Example usage with a p-value threshold (for example, 0.05)
results = run_inference(model=model, 
              image_path="Cirrhosis_T2_3D/test_images/", 
              label_path="Cirrhosis_T2_3D/test_masks/", 
              threshold=0.1,
              mode="avg",
              results=results)
results = run_inference(model=model, 
              image_path="Cirrhosis_T2_3D/test_images/", 
              label_path="Cirrhosis_T2_3D/test_masks/", 
              threshold=0.05,
              mode="stat",
              results=results)
results = run_inference(model=model, 
              image_path="Cirrhosis_T2_3D/test_images/", 
              label_path="Cirrhosis_T2_3D/test_masks/", 
              threshold=0.1,
              mode="stat",
              results=results)



df = pd.DataFrame.from_dict(results, orient="index")
df.index.name = "MRI Path | Mode | Threshold"
df.to_csv("results/scores.csv")