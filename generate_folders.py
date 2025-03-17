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
model.eval()




def save_heatmap(image, save_path):
    """Save a 2D heatmap as a JPG image."""
    fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=100) 
    ax.imshow(image.T, cmap='jet', interpolation='nearest', vmin = 0, vmax = 1)
    ax.axis('off')  # Remove axes for cleaner images
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)

def run_inference(model, image_path, label_path, threshold, mode, results):
    # Define cropping/padding transformation to standard size
    cropping = tio.CropOrPad((320, 320, 32), padding_mode=0)
    
    for img_filename in tqdm(os.listdir(image_path)):
        base_path = os.path.join("results/generated_" + str(mode) + str(int(threshold * 100)), img_filename[:-7])
        os.makedirs(base_path, exist_ok=True)
        
        # Load and process image
        nii_img = nib.load(os.path.join(image_path, img_filename))
        affine = nii_img.affine
        img_data = nii_img.get_fdata()
        img_tensor = torch.tensor(img_data).unsqueeze(0)
        img_tensor = cropping(img_tensor).float().squeeze().numpy()
        nii_img_cropped = nib.Nifti1Image(img_tensor, affine)
        nib.save(nii_img_cropped, os.path.join(base_path, "mri.nii.gz"))
        img_tensor = torch.tensor(img_tensor).to(device)
        with torch.no_grad():
            outputs = model(img_tensor.unsqueeze(0).unsqueeze(0))  # outputs shape: (1, n_channels, H, W, D)
            n_channels = outputs.shape[1]
            #outputs = outputs.cpu().detach().numpy()
            if mode == "stat":
                # Compute mean and std across channels for each voxel
                mean_voxel = outputs.mean(dim=1).squeeze().cpu().detach().numpy()  # shape: (1, H, W, D)
                std_voxel = outputs.std(dim=1).squeeze().cpu().detach().numpy()    # shape: (1, H, W, D)
                # Avoid division by zero by adding a small epsilon
                epsilon = 1e-8
                # Compute t-statistic: (mean - 1) / (std / sqrt(n_channels))
                t_stat = (mean_voxel - 1.0) / (std_voxel / (n_channels ** 0.5) + epsilon)
                # Convert t_stat to numpy array for SciPy
                # t_stat = t_stat  # shape: (H, W, D)
                
                # Compute one-sided p-value (probability that mean > 1)
                # degrees of freedom = n_channels - 1
                p_value = 2 * (1 - stats.t.cdf(t_stat, df=n_channels - 1))

                #print(p_value.min(), p_value.mean(), p_value.std(), p_value.max())
                mean_mean = mean_voxel.mean()
                mean_std = mean_voxel.std()
                std_mean = std_voxel.mean()
                std_std = std_voxel.std()
                pval_mean = t_stat.mean()
                pval_std = t_stat.std()
                mean_voxel = (mean_voxel - np.min(mean_voxel))/(np.max(mean_voxel) - np.min(mean_voxel) + 1e-8)
                std_voxel = (std_voxel - np.min(std_voxel))/(np.max(std_voxel) - np.min(std_voxel) + 1e-8)
                t_stat = (t_stat - np.min(t_stat))/(np.max(t_stat) - np.min(t_stat) + 1e-8)
                for d in range(outputs.shape[4]):
                    mean_slice = mean_voxel[ :, :, d]  # Shape: (H, W)
                    std_slice = std_voxel[ :, :, d]    # Shape: (H, W)
                    pval_slice = t_stat[ :, :, d]     # Shape: (H, W)
                    #print(mean_slice.shape)
                    os.makedirs(os.path.join(base_path, "heatmaps_mean"), exist_ok=True)
                    os.makedirs(os.path.join(base_path, "heatmaps_std"), exist_ok=True)
                    os.makedirs(os.path.join(base_path, "heatmaps_pval"), exist_ok=True)
                    save_heatmap(mean_slice, os.path.join(base_path, "heatmaps_mean", f"mean_{d}.jpg"))
                    save_heatmap(std_slice, os.path.join(base_path, "heatmaps_std", f"std_{d}.jpg"))
                    save_heatmap(pval_slice, os.path.join(base_path, "heatmaps_pval", f"pval_{d}.jpg"))
            elif mode == "avg":
                p_value = outputs.mean(dim=1).squeeze().cpu().detach().numpy()
                mean_mean, mean_std, std_mean, std_std, pval_mean, pval_std = [None] * 6
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

        #print(seg.shape, label_data_cropped.shape)
        preds = torch.tensor(seg).unsqueeze(0).unsqueeze(0).to(device)
        target = torch.tensor(label_data_cropped).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            jaccard = JaccardIndex(preds, target).item()
            dice = Dice(preds, target).item()
            precision = Precision(preds, target).item()
            recall = Recall(preds, target).item()
            #assd = ASSD(preds, target).item()
            #hd95 = HD95(preds, target).item()
        
        row_name = f"{img_filename} | mode: {mode}, threshold: {threshold}"
        
        results[row_name] = {
            "mode": mode,
            "threshold": threshold,
            "Jaccard Index": jaccard,
            "Dice": dice,
            "Precision": precision,
            "Recall": recall,
            "Mean of Mean": mean_mean,
            "Std of Mean": mean_std,
            "Mean of Std": std_mean,
            "Std of Std": std_std,
            "Mean of Pval": pval_mean,
            "Std of Pval": pval_std,
            #"ASSD": assd,
            #"HD95": hd95
        }

        # Compute per-channel metrics as separate rows
        if mode == "avg":
            for ch in range(n_channels):
                single_channel = outputs[:, ch, :, :, :].squeeze().cpu().detach().numpy()
                single_channel_seg = (single_channel > threshold).astype(np.float32)

                preds_ch = torch.tensor(single_channel_seg).unsqueeze(0).unsqueeze(0).to(device)
                
                jaccard_ch = JaccardIndex(preds_ch, target).item()
                dice_ch = Dice(preds_ch, target).item()
                precision_ch = Precision(preds_ch, target).item()
                recall_ch = Recall(preds_ch, target).item()

                row_name_ch = f"{img_filename}_Ch{ch} | mode: {mode}, threshold: {threshold}"
                results[row_name_ch] = {
                    "mode": mode,
                    "threshold": threshold,
                    "Jaccard Index": jaccard_ch,
                    "Dice": dice_ch,
                    "Precision": precision_ch,
                    "Recall": recall_ch,
                    "Mean of Mean": None,
                    "Std of Mean": None,
                    "Mean of Std": None,
                    "Std of Std": None,
                    "Mean of Pval": None,
                    "Std of Pval": None
                }
                save_path = os.path.join(base_path, "avg")
                os.makedirs(save_path, exist_ok=True)  # Ensure directory exists
                nii_img = nib.Nifti1Image(single_channel_seg, affine)
                nib.save(nii_img, os.path.join(save_path, f"{ch}.nii.gz"))
    return results

results = {}
# Example usage with a p-value threshold (for example, 0.05)
results = run_inference(model=model, 
              image_path="Cirrhosis_T2_3D/test_images/", 
              label_path="Cirrhosis_T2_3D/test_masks/", 
              threshold=0.5,
              mode="avg",
              results=results)
results = run_inference(model=model, 
              image_path="Cirrhosis_T2_3D/test_images/", 
              label_path="Cirrhosis_T2_3D/test_masks/", 
              threshold=0.75,
              mode="avg",
              results=results)
results = run_inference(model=model, 
              image_path="Cirrhosis_T2_3D/test_images/", 
              label_path="Cirrhosis_T2_3D/test_masks/", 
              threshold=0.01,
              mode="stat",
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
