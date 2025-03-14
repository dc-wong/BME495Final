import os
import torch
import torchio as tio
import torch.nn as nn
import nibabel as nib
from tqdm import tqdm
import numpy as np
import scipy.stats as stats

from loss import MultiLoss
from model import SingleModel

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

def run_inference(model, image_path, label_path, p_threshold):
    """
    For each image, this function:
      - Applies a cropping transformation.
      - Runs inference to get a multi-channel output.
      - Computes, voxel-wise, the mean and standard deviation across channels.
      - Performs a one-sided t-test (null hypothesis: mean == 1) to compute a p-value
        for the probability that the voxel's value is greater than 1.
      - Thresholds the p-value at p_threshold to create a binary mask.
      - Saves the resulting mask and the corresponding label.
    
    Parameters:
      model: The segmentation model.
      image_path: Path to the directory with test images.
      label_path: Path to the directory with test labels.
      p_threshold: p-value threshold for segmentation (e.g., 0.05).
    """
    # Define cropping/padding transformation to standard size
    cropping = tio.CropOrPad((320, 320, 32), padding_mode=0)
    
    for img_filename in tqdm(os.listdir(image_path)):
        base_path = os.path.join("generated_pval_" + str(p_threshold), img_filename[:-7])
        os.makedirs(base_path, exist_ok=True)
        
        # Load and process image
        nii_img = nib.load(os.path.join(image_path, img_filename))
        affine = nii_img.affine
        img_data = nii_img.get_fdata()
        img_tensor = torch.tensor(img_data).unsqueeze(0)
        img_tensor = cropping(img_tensor).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)  # outputs shape: (1, n_channels, H, W, D)
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
            p_value = stats.t.pdf(t_stat, df=n_channels - 1)
            print(p_value.min(), p_value.mean(), p_value.std(), p_value.max())
            # Create segmentation mask: 1 if p-value is below the threshold, else 0
            seg = (p_value > p_threshold).astype(np.float32)
            
            # Save the segmentation mask as a NIfTI file
            nii_seg = nib.Nifti1Image(seg, affine)
            nib.save(nii_seg, os.path.join(base_path, "segmentation.nii.gz"))
        
        # Process the label similarly (only cropping here)
        nii_label = nib.load(os.path.join(label_path, img_filename))
        affine_label = nii_label.affine
        label_data = nii_label.get_fdata()
        label_tensor = torch.tensor(label_data).unsqueeze(0)
        label_data_cropped = cropping(label_tensor).squeeze().numpy()
        nii_label_cropped = nib.Nifti1Image(label_data_cropped, affine_label)
        nib.save(nii_label_cropped, os.path.join(base_path, "original.nii.gz"))

# Example usage with a p-value threshold (for example, 0.05)
run_inference(model=model, 
              image_path="Cirrhosis_T2_3D/test_images/", 
              label_path="Cirrhosis_T2_3D/test_masks/", 
              p_threshold=0.20)
