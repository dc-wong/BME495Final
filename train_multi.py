### load the torch and torchvision libraries
import os
import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import torch._dynamo
# torch._dynamo.config.compile_backend = "nvfuser" 

from loss import MultiLoss
from model import SingleModel
from datasets import TransformedDataset
from metrics import MultiAccuracy
import torchio as tio

import os
from dotenv import load_dotenv
load_dotenv(".env", override=True)

# from torch.utils.tensorboard import SummaryWriter


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
model.to(device)



test_image = torch.rand(4, 1, 320, 320, 32).to(device)
output = model(test_image)
print(output.shape)
print(output.min(), output.max())
del output
torch.cuda.empty_cache()

transform = tio.Compose([
    #tio.ZNormalization(include=['image']),  # Normalize intensity only on the image
    tio.RandomAffine(scales=(0.9, 1.1), degrees=10, translation=(10, 10, 10), include=['image','label']),  # Apply only to the image
    tio.RandomElasticDeformation(num_control_points=8, max_displacement=4, locked_borders=2, p=0.5, include=['image','label']),  # Elastic transform
    tio.RandomBiasField(coefficients = 0.5, order = 3),
    #tio.RandomFlip(axes=(0, 1, 2), p=0.5, include=['image','label'])  # Flip along random axes
])

trainset = TransformedDataset(image_dir = os.path.abspath("Cirrhosis_T2_3D/train_images"), label_dir = os.path.abspath("Cirrhosis_T2_3D/train_masks"),  transform=transform)
# testset = TransformedDataset(image_dir = r"C:\PROJECTS\BME495\FinalProject\Cirrhosis_T2_3D\test_images", label_dir = r"C:\PROJECTS\BME495\FinalProject\Cirrhosis_T2_3D\test_masks",  transform=transform)
# test_loader = torch.utils.data.DataLoader(testset, batch_size = 1, shuffle = True) # batch size is 1 because of varying sizes of MRI
valset = TransformedDataset(image_dir = os.path.abspath("Cirrhosis_T2_3D/valid_images"), label_dir = os.path.abspath("Cirrhosis_T2_3D/valid_masks"),  transform=transform)

#sampler = tio.LabelSampler(patch_size = (128, 128, 16), label_name='label', label_probabilities={0:0.05,1:0.95})

#train_queue = tio.Queue(subjects_dataset=trainset, max_length=1200, samples_per_volume=4, sampler=sampler, shuffle_subjects=True, shuffle_patches=True)
#val_queue = tio.Queue(subjects_dataset=valset, max_length=300, samples_per_volume=4, sampler=sampler, shuffle_subjects=True, shuffle_patches=True)
#train_loader = torch.utils.data.DataLoader(train_queue, batch_size = 4)
#train_loader = tio.SubjectsLoader(train_queue, batch_size = 16, shuffle = True)
#val_loader = torch.utils.data.DataLoader(val_queue, batch_size = 4)
#val_loader = tio.SubjectsLoader(val_queue, batch_size = 16, shuffle = True)
train_loader = tio.SubjectsLoader(trainset, batch_size = 4, shuffle = True) # batch size is 1 because of varying sizes of MRI
val_loader = tio.SubjectsLoader(valset, batch_size = 4, shuffle = True) # batch size is 1 because of varying sizes of MRI



dataloaders = {"train": train_loader, "val": val_loader}

# just to make sure things are working correctly
for batch in train_loader:
    inputs = batch['image'][tio.DATA]  # Extract tensor from Subject
    labels = batch['label'][tio.DATA] # Extract tensor from Subject
    print("Image batch shape:", inputs.shape)
    print("Image batch range:", inputs.min(), inputs.max())
    print("Label batch shape:", labels.shape)
    print("Label batch unique values:", labels.unique())
    break


def train_model(model, criterion, optimizer, num_epochs=25, continue_training = False): #scheduler
    # Create a temporary directory to save training checkpoints
    best_model_params_path = os.path.join('weights', 'base_3d_unet.pt')
    if continue_training:
        model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
    best_acc = 0.0
    torch.save(model.state_dict(), best_model_params_path)
    scaler = torch.amp.GradScaler("cuda")
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0

            # zero the parameter gradients
            # optimizer.zero_grad()

            # # Iterate over data.
            # accumulation_steps = 10
            for batch_idx, batch in enumerate(dataloaders[phase]):
                # inputs = inputs.to(device)
                # labels = labels.to(device)
                inputs = batch['image'][tio.DATA].float().to(device)  # Extract tensor from Subject
                labels = batch['label'][tio.DATA].float().to(device)  # Extract tensor from Subject
                
                # # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    with torch.autocast(device_type="cuda", dtype=torch.float16):

                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        scaler.scale(loss).backward()  # Use scaled gradients
                            # loss.backward()
                            # optimizer.step()
                            # if (batch_idx + 1) % accumulation_steps == 0:
                            #     optimizer.step()  # Only step optimizer every X steps
                        #     # zero the parameter gradients
                        scaler.step(optimizer)  # Step optimizer
                        scaler.update()
                            #     optimizer.step()
                            #     optimizer.zero_grad()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                with torch.no_grad():
                    running_corrects += MultiAccuracy(outputs, labels.data)
                del inputs, labels, loss, outputs
                torch.cuda.empty_cache()  # Free unused GPU memory

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            #if phase == 'val':
                #scheduler.step(epoch_loss)
                #print(scheduler.get_last_lr())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_params_path)
            del epoch_loss, running_loss, running_corrects #, epoch_acc

        print()

    print(f'Best val acc: {best_acc:4f}')
    # load best model weights
    model.load_state_dict(torch.load(best_model_params_path))
    return model


### Why cross entropy loss?
criterion = MultiLoss()
### SGD, Adam, RMSprop, ... AdamW
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
### decay learning rate
#exp_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, threshold= 1e-3, factor=0.999)

model = train_model(model, criterion, optimizer, num_epochs=250, continue_training=False) # exp_lr_scheduler
