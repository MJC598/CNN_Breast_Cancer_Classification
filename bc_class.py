import torch, pandas as pd, numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from skimage import transform

import os
import time
import copy
import torchvision
from torchvision import transforms, utils, models


class CustomImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.bc_frame = pd.read_csv(csv_file, sep=" ", header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.bc_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, str((self.bc_frame.iloc[idx, 0].split("/")[1]).split(".")[0]) + ' resized.png')
        if not os.path.isfile(img_name):
            print("file: " + img_name + " does not exist!\n")
            return None
        image = Image.open(img_name).convert('RGB')
        classes_txt = self.bc_frame.iloc[idx, 0].split(",")[0]
        classes = (np.array([0, 1]) if classes_txt == 'MALIGNANT' else np.array([1, 0]))
        if self.transform:
            image = self.transform(image)
            classes = torch.from_numpy(classes)
        sample = {'image': image, 'classes': classes}

        return sample

"""
Function for training the model
:param: model - model architecture to be trained
:param: criterion - loss function
:param: optimizer - optimizer scheme (Stochastic Gradient Descent)
:param: scheduler - drop off scheduler
:param: num_epochs - epochs (default 25)

:output: trained model
"""
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for samples in dataloaders[phase]:
                inputs = samples['image']
                labels = samples['classes']
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    _, ans = torch.max(labels, 1)
                    loss = criterion(outputs, ans)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == ans)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            if phase == 'val':
                val_loss[epoch] = epoch_loss
                val_acc[epoch] = epoch_acc
            else:
                train_loss[epoch] = epoch_loss
                train_acc[epoch] = epoch_acc

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model

"""
the transformations for validation and training datasets
"""
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
image_datasets = {x: CustomImageDataset(csv_file='/home/matt/data/Breast_Cancer_Data/Resized-ROI-PNG-Images/index.csv', 
                               root_dir='/home/matt/data/Breast_Cancer_Data/Resized-ROI-PNG-Images/',
                               transform=data_transforms[x])
                     for x in ['train', 'val']}

# image_datasets = {x: CustomImageDataset(csv_file='/home/matt/data/Breast_Cancer_Data/ResizedPNGImages/index.csv', 
#                                root_dir='/home/matt/data/Breast_Cancer_Data/ResizedPNGImages/',
#                                transform=data_transforms[x])
#                      for x in ['train', 'val']}


dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = ['benign', 'malignant']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sample = next(iter(dataloaders['train']))
inputs = sample['image']
classes = sample['classes']
c = []
for vec in classes:
    if torch.equal(vec, torch.tensor([0, 1])):
        c.append(1)
    else:
        c.append(0)
out = torchvision.utils.make_grid(inputs)


model_ft = models.resnet50(pretrained=True)
for param in model_ft.parameters():
    param.requires_grad = False

num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

# model_ft.load_state_dict(torch.load("Nov19_r50_5ep_classifier.pt"))
# for param in model_ft.parameters():
    # param.requires_grad = True

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0001, momentum=0.75)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.01)

epochs = 100

train_acc = [0 for i in range(epochs)]
train_loss = [0 for i in range(epochs)]
val_acc = [0 for i in range(epochs)]
val_loss = [0 for i in range(epochs)]

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=epochs)
torch.save(model_ft.state_dict(), "Dec16_r50_100ep_roi.pt")

fig = plt.figure(figsize=(16, 4))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
ax1.title.set_text('Training Accuracy')
ax1.plot(train_acc)
ax2.title.set_text('Training Loss')
ax2.plot(train_loss)
ax3.title.set_text('Validation Accuracy')
ax3.plot(val_acc)
ax4.title.set_text('Validation Loss')
ax4.plot(val_loss)
plt.show()
