# Multi-Headed U-Net

This project trains a multi-headed 3D U-net on the CirrMRI600+ dataset for a segmentation task. The multi-headed output of the model allows for improved segmentation by incorporating information about variance in output that arises due to the random initialization of the model weights without major changes to the model architecture and without greatly increased computational requirements.

## Dataset

Download the CirrMRI600+ dataset from [https://osf.io/cuk24/](https://osf.io/cuk24/).

After it has been downloaded, extract the data from the "Cirrhosis_T2_3D.zip" file and save it in this folder. This is that data that will be used for this model.

## Training

To train the model, run the following:

```
python train.py
```

This will train the model for 250 epochs at a learning rate of 1e-4 with an Adam optimizer. After training, weights will be stored in the "weights" folder.

## Inference

To use the model, run the following:

```
python inference.py /path/to/image.nii.gz mode
```

Where "/path/to/image.nii.gz" is the path to the MRI that you would like to be segmented and mode is either "avg" or "var".
