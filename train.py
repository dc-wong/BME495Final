### load the torch packges
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

### load torchio for train time augmentations
import torchio as tio

### import requisite classes and functions from other files
from loss import MultiLoss
from model import SingleModel
from datasets import TransformedDataset
from metrics import MultiAccuracy

### check torch and cuda version
print(torch.__version__)
print("cuda" if torch.cuda.is_available() else "cpu")
print(torch.version.cuda)  # Check CUDA version
print(torch.backends.cudnn.enabled)

### load the model into the device
model = SingleModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


### run a random input of the same size as the input to check
### that the output is the correct shape and that you do not run out of memory
test_image = torch.rand(1, 1, 320, 320, 32).to(device)
output = model(test_image)
print(output.shape)
### clear to save ram
del test_image, output
torch.cuda.empty_cache()

### the transformations that will be used
transform = tio.Compose([
    ### Apply a random bias field on the MRI
    tio.RandomBiasField(coefficients = 0.5, order = 3, include=['image']),
    ### Apply random rigid transformations (rotations, translations) on both image and mask
    tio.RandomAffine(scales=(0.9, 1.1), degrees=10, translation=(10, 10, 10), include=['image','label']),  
    ### Apply random elastic deformations to both image and mask
    tio.RandomElasticDeformation(num_control_points=8, max_displacement=4, locked_borders=2, p=0.5, include=['image','label']),  
])

### Load both the training and validation sets
trainset = TransformedDataset(image_dir = os.path.abspath("Cirrhosis_T2_3D/train_images"), label_dir = os.path.abspath("Cirrhosis_T2_3D/train_masks"),  transform=transform)
valset = TransformedDataset(image_dir = os.path.abspath("Cirrhosis_T2_3D/valid_images"), label_dir = os.path.abspath("Cirrhosis_T2_3D/valid_masks"),  transform=transform)
### batch size of 1 can be increased since inputs are cropped to the same size
train_loader = tio.SubjectsLoader(trainset, batch_size = 1, shuffle = True) 
val_loader = tio.SubjectsLoader(valset, batch_size = 1, shuffle = True) 
dataloaders = {"train": train_loader, "val": val_loader}

### verify data has correct information
for batch in train_loader:
    inputs = batch['image'][tio.DATA]  
    labels = batch['label'][tio.DATA] 
    ### check that the inputs have the correct shape and
    ### that the minimum and maxiumum are reasonable
    print("Image batch shape:", inputs.shape)
    print("Image batch range:", inputs.min(), inputs.max())
    ### check that the labels have the correct shape and
    ### that the values are correct (should be 0 and 1)
    print("Label batch shape:", labels.shape)
    print("Label batch unique values:", labels.unique())
    break

### create the training loop that returns the trained model
def train_model(model, criterion, optimizer, num_epochs=25):
    best_acc = 0.0
    # Create a directory to save training checkpoints
    best_model_params_path = os.path.join('weights', 'model_weights.pth')
    torch.save(model.state_dict(), best_model_params_path)

    ### start the training loop
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        ### Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            ### change the mode of the 
            if phase == 'train':
                model.train() 
            else:
                model.eval()   
            running_loss = 0.0
            running_corrects = 0

            ### Iterate over data.
            for batch_idx, batch in enumerate(dataloaders[phase]):
                ### load inputs and masks to the device
                inputs = batch['image'][tio.DATA].float().to(device)  
                labels = batch['label'][tio.DATA].float().to(device)
                
                ### zero the parameter gradients
                optimizer.zero_grad()

                ### forward pass
                ### track gradients if we are in the train phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    ### backward pass if we are training
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                ### stats
                ### no_grad to save memory
                with torch.no_grad():
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += MultiAccuracy(outputs, labels.data)
                    
                ### free memory
                del inputs, labels, loss, outputs
                torch.cuda.empty_cache() 

            ### loss and accuracy over the entire epoch
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            ### save the best version of the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_loss
                torch.save(model.state_dict(), best_model_params_path)

            ### delete to save memory
            del epoch_loss, running_loss, running_corrects 
        print()
    print(f'Best val acc: {best_acc:4f}')
    
    ### load best model weights
    model.load_state_dict(torch.load(best_model_params_path))
    return model


### load the loss function
criterion = MultiLoss()
### use the Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-2)
### Train the model
model = train_model(model, criterion, optimizer, num_epochs=250) 
