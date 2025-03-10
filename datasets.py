import os
import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib

import torchio as tio
from torchio import SubjectsDataset, SubjectsLoader

class TransformedDataset(tio.SubjectsDataset):
    def __init__(self, image_dir, label_dir, target_shape = (400, 400, 32), transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.label_files = sorted(os.listdir(label_dir))
        self.transform = transform
        self.target_shape = target_shape

        assert len(self.image_files) == len(self.label_files), "Mismatch between images and labels"

        subjects = []
        for img_file, label_file in zip(self.image_files, self.label_files):
            img_path = os.path.join(self.image_dir, img_file)
            label_path = os.path.join(self.label_dir, label_file)

            subject = tio.Subject(
                image=tio.ScalarImage(img_path),
                label=tio.LabelMap(label_path)
            )
            if self.transform:
                subject = self.transform(subject)
            #affine = np.eye(4)
            #affine[:3,:3] = np.diag(subject['image'].spacing)
            #affine[:3,3] = subject['image'].affine[:3, 3]
            #resample_transform = tio.Resample(target = (self.target_shape, affine), image_interpolation = 'bspline', label_interpolation = 'label_gaussian')
            #subject['image'] = resample_transform(subject['image'])
            #resample_transform_label = tio.Resample(target = subject['image'], label_interpolation = 'nearest')
            #subject['label'] = resample_transform(subject['label'])
            #print(subject['label'].data.unique())
            #subject['label'].set_data((subject['label'].data > 0.5).float())
            #print(subject['image'].shape, subject['label'].shape)
            #assert torch.equal(subject['label'].data.unique(), torch.tensor([0,1]).float()), "Labels missing labels 3"
            #assert subject['image'].shape == (1, 400, 400, 32), "Mismatch shape after transform"
            subjects.append(subject)

        super().__init__(subjects)  # Initialize properly

    def __getitem__(self, idx):
        subject = super().__getitem__(idx)  # Ensure transformation returns a `tio.Subject`)
        return subject
