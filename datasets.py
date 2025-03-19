import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torchio as tio
from torchio import SubjectsDataset, SubjectsLoader

class TransformedDataset(tio.SubjectsDataset):
    def __init__(self, image_dir, label_dir, target_shape = (320, 320, 32), transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.label_files = sorted(os.listdir(label_dir))
        self.transform = transform
        self.cropOrPad = tio.CropOrPad(target_shape, padding_mode = 0, include  = ('image', 'label'))
        self.target_shape = target_shape

        ### ensure that the files are all detected
        assert len(self.image_files) == len(self.label_files), "Mismatch between images and labels"

        subjects = []
        ### load all the subjects and store them
        for img_file, label_file in tqdm(zip(self.image_files, self.label_files)):
            img_path = os.path.join(self.image_dir, img_file)
            label_path = os.path.join(self.label_dir, label_file)
            subject = tio.Subject(
                image=tio.ScalarImage(img_path),
                label=tio.LabelMap(label_path)
            )
            ### crop the images to 320x320x32
            subject = self.cropOrPad(subject)
            subjects.append(subject)
        super().__init__(subjects)  # Initialize properly

    def __getitem__(self, idx):
        ### get the subect
        subject = super().__getitem__(idx)
        ### conduct the random transformations on the subject
        subject = self.transform(subject)
        return subject
