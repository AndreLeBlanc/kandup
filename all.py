# %% [code] {"execution":{"iopub.status.busy":"2021-07-08T10:47:31.092822Z","iopub.execute_input":"2021-07-08T10:47:31.093176Z","iopub.status.idle":"2021-07-08T10:47:45.126047Z","shell.execute_reply.started":"2021-07-08T10:47:31.093145Z","shell.execute_reply":"2021-07-08T10:47:45.125255Z"}}
### Unzipping Dataset
import zipfile

with zipfile.ZipFile("input/train.zip","r") as z:
    z.extractall(".")

with zipfile.ZipFile("input/test.zip","r") as z:
    z.extractall(".")

# %% [code] {"execution":{"iopub.status.busy":"2021-07-08T10:47:45.127913Z","iopub.execute_input":"2021-07-08T10:47:45.128277Z","iopub.status.idle":"2021-07-08T10:47:45.139778Z","shell.execute_reply.started":"2021-07-08T10:47:45.128248Z","shell.execute_reply":"2021-07-08T10:47:45.138857Z"}}
import os
import cv2
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
from torchvision.utils import make_grid
import alexnet

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
vals = []

DIR_TRAIN = "data/train/"
DIR_TEST = "data/test"

# %% [code] {"execution":{"iopub.status.busy":"2021-07-08T10:47:45.141879Z","iopub.execute_input":"2021-07-08T10:47:45.142265Z","iopub.status.idle":"2021-07-08T10:47:45.184925Z","shell.execute_reply.started":"2021-07-08T10:47:45.142228Z","shell.execute_reply":"2021-07-08T10:47:45.184137Z"}}
### Checking Data Format
imgs = os.listdir(DIR_TRAIN)
test_imgs = os.listdir(DIR_TEST)

print(imgs[:5])
print(test_imgs[:5])

# %% [code] {"execution":{"iopub.status.busy":"2021-07-08T10:47:45.186783Z","iopub.execute_input":"2021-07-08T10:47:45.187145Z","iopub.status.idle":"2021-07-08T10:47:45.223169Z","shell.execute_reply.started":"2021-07-08T10:47:45.187102Z","shell.execute_reply":"2021-07-08T10:47:45.222346Z"}}
### Class Distribution
dogs_list = [img for img in imgs if img.split(".")[0] == "dog"]
cats_list = [img for img in imgs if img.split(".")[0] == "cat"]

print("No of Dogs Images: ",len(dogs_list))
print("No of Cats Images: ",len(cats_list))

class_to_int = {"dog" : 0, "cat" : 1}
int_to_class = {0 : "dog", 1 : "cat"}

# %% [code] {"execution":{"iopub.status.busy":"2021-07-08T10:47:45.22662Z","iopub.execute_input":"2021-07-08T10:47:45.226972Z","iopub.status.idle":"2021-07-08T10:47:45.232153Z","shell.execute_reply.started":"2021-07-08T10:47:45.226925Z","shell.execute_reply":"2021-07-08T10:47:45.231247Z"}}
### Transforms for image - ToTensor and other augmentations
def get_transform():
    return T.Compose([T.ToTensor()])


# %% [code] {"execution":{"iopub.status.busy":"2021-07-08T10:47:45.234612Z","iopub.execute_input":"2021-07-08T10:47:45.23504Z","iopub.status.idle":"2021-07-08T10:47:45.250447Z","shell.execute_reply.started":"2021-07-08T10:47:45.235Z","shell.execute_reply":"2021-07-08T10:47:45.249487Z"}}
### Dataset Class - for retriving images and labels
class CatDogDataset(Dataset):

    def __init__(self, imgs, class_to_int, mode = "train", transforms = None):

        super().__init__()
        self.imgs = imgs
        self.class_to_int = class_to_int
        self.mode = mode
        self.transforms = transforms

    def __getitem__(self, idx):

        image_name = self.imgs[idx]

        ### Reading, converting and normalizing image
        img = cv2.imread(DIR_TRAIN + image_name, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224,224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img /= 255.

        if self.mode == "train" or self.mode == "val":

            ### Preparing class label
            label = self.class_to_int[image_name.split(".")[0]]
            label = torch.tensor(label, dtype = torch.float32)

            ### Apply Transforms on image
            img = self.transforms(img)

            return img, label

        elif self.mode == "test":

            ### Apply Transforms on image
            img = self.transforms(img)

            return img


    def __len__(self):
        return len(self.imgs)


# %% [code] {"execution":{"iopub.status.busy":"2021-07-08T10:47:45.252876Z","iopub.execute_input":"2021-07-08T10:47:45.253301Z","iopub.status.idle":"2021-07-08T10:47:45.273193Z","shell.execute_reply.started":"2021-07-08T10:47:45.253265Z","shell.execute_reply":"2021-07-08T10:47:45.272208Z"}}
### Splitting data into train and val sets
train_imgs, val_imgs = train_test_split(imgs, test_size = 0.25)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-08T10:47:45.275242Z","iopub.execute_input":"2021-07-08T10:47:45.275633Z","iopub.status.idle":"2021-07-08T10:47:45.288002Z","shell.execute_reply.started":"2021-07-08T10:47:45.275595Z","shell.execute_reply":"2021-07-08T10:47:45.286865Z"}}
### Dataloaders
train_dataset = CatDogDataset(train_imgs, class_to_int, mode = "train", transforms = get_transform())
val_dataset = CatDogDataset(val_imgs, class_to_int, mode = "val", transforms = get_transform())
test_dataset = CatDogDataset(test_imgs, class_to_int, mode = "test", transforms = get_transform())

train_data_loader = DataLoader(
    dataset = train_dataset,
    num_workers = 4,
    batch_size = 156,
    shuffle = True
)

val_data_loader = DataLoader(
    dataset = val_dataset,
    num_workers = 4,
    batch_size = 156,
    shuffle = True
)

test_data_loader = DataLoader(
    dataset = test_dataset,
    num_workers = 4,
    batch_size = 16,
    shuffle = True
)


# %% [code] {"execution":{"iopub.status.busy":"2021-07-08T10:47:45.290081Z","iopub.execute_input":"2021-07-08T10:47:45.290573Z","iopub.status.idle":"2021-07-08T10:47:49.65355Z","shell.execute_reply.started":"2021-07-08T10:47:45.290533Z","shell.execute_reply":"2021-07-08T10:47:49.64912Z"}}
### Visualize Random Images from Train set
for images, labels in train_data_loader:

    fig, ax = plt.subplots(figsize = (10, 10))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(make_grid(images, 4).permute(1,2,0))
    break


# %% [code] {"execution":{"iopub.status.busy":"2021-07-08T10:47:49.65561Z","iopub.execute_input":"2021-07-08T10:47:49.655988Z","iopub.status.idle":"2021-07-08T10:47:49.661647Z","shell.execute_reply.started":"2021-07-08T10:47:49.655934Z","shell.execute_reply":"2021-07-08T10:47:49.660792Z"}}
### GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# %% [code] {"execution":{"iopub.status.busy":"2021-07-08T10:47:49.663511Z","iopub.execute_input":"2021-07-08T10:47:49.663912Z","iopub.status.idle":"2021-07-08T10:47:49.673079Z","shell.execute_reply.started":"2021-07-08T10:47:49.663871Z","shell.execute_reply":"2021-07-08T10:47:49.672232Z"}}
### Function to calculate accuracy
def accuracy(preds, trues):

    ### Converting preds to 0 or 1
    preds = [1 if preds[i] >= 0.5 else 0 for i in range(len(preds))]

    ### Calculating accuracy by comparing predictions with true labels
    acc = [1 if preds[i] == trues[i] else 0 for i in range(len(preds))]

    ### Summing over all correct predictions
    acc = np.sum(acc) / len(preds)

    return (acc * 100)



# %% [code] {"execution":{"iopub.status.busy":"2021-07-08T10:47:49.674427Z","iopub.execute_input":"2021-07-08T10:47:49.67554Z","iopub.status.idle":"2021-07-08T10:47:49.689947Z","shell.execute_reply.started":"2021-07-08T10:47:49.675502Z","shell.execute_reply":"2021-07-08T10:47:49.689106Z"}}
### Function - One Epoch Train
def train_one_epoch(train_data_loader):

    ### Local Parameters
    epoch_loss = []
    epoch_acc = []
    start_time = time.time()

    ###Iterating over data loader
    for images, labels in train_data_loader:

        #Loading images and labels to device
        images = images.to(device)
        labels = labels.to(device)
        labels = labels.reshape((labels.shape[0], 1)) # [N, 1] - to match with preds shape

        #Reseting Gradients
        optimizer.zero_grad()

        #Forward
        preds = model(images)

        #Calculating Loss
        _loss = criterion(preds, labels)
        loss = _loss.item()
        epoch_loss.append(loss)

        #Calculating Accuracy
        acc = accuracy(preds, labels)
        epoch_acc.append(acc)

        #Backward
        _loss.backward()
        optimizer.step()

    ###Overall Epoch Results
    end_time = time.time()
    total_time = end_time - start_time

    ###Acc and Loss
    epoch_loss = np.mean(epoch_loss)
    epoch_acc = np.mean(epoch_acc)

    ###Storing results to logs
    train_logs["loss"].append(epoch_loss)
    train_logs["accuracy"].append(epoch_acc)
    train_logs["time"].append(total_time)

    return epoch_loss, epoch_acc, total_time


# %% [code] {"execution":{"iopub.status.busy":"2021-07-08T10:47:49.691882Z","iopub.execute_input":"2021-07-08T10:47:49.692267Z","iopub.status.idle":"2021-07-08T10:47:49.70605Z","shell.execute_reply.started":"2021-07-08T10:47:49.692232Z","shell.execute_reply":"2021-07-08T10:47:49.705191Z"}}
### Function - One Epoch Valid
def val_one_epoch(val_data_loader):

    ### Local Parameters
    epoch_loss = []
    epoch_acc = []
    start_time = time.time()

    ###Iterating over data loader
    for images, labels in val_data_loader:

        #Loading images and labels to device
        images = images.to(device)
        labels = labels.to(device)
        labels = labels.reshape((labels.shape[0], 1)) # [N, 1] - to match with preds shape

        #Forward
        preds = model(images)

        #Calculating Loss
        _loss = criterion(preds, labels)
        loss = _loss.item()
        epoch_loss.append(loss)

        #Calculating Accuracy
        acc = accuracy(preds, labels)
        epoch_acc.append(acc)

    ###Overall Epoch Results
    end_time = time.time()
    total_time = end_time - start_time

    ###Acc and Loss
    epoch_loss = np.mean(epoch_loss)
    epoch_acc = np.mean(epoch_acc)

    ###Storing results to logs
    val_logs["loss"].append(epoch_loss)
    val_logs["accuracy"].append(epoch_acc)
    val_logs["time"].append(total_time)

    return epoch_loss, epoch_acc, total_time


# %% [code] {"execution":{"iopub.status.busy":"2021-07-08T10:47:49.707944Z","iopub.execute_input":"2021-07-08T10:47:49.708398Z","iopub.status.idle":"2021-07-08T10:47:50.422715Z","shell.execute_reply.started":"2021-07-08T10:47:49.708363Z","shell.execute_reply":"2021-07-08T10:47:50.421759Z"}}
### VGG16 Pretrained Model
import cnnGabZern
#model = cnnGabZern.AlexNet("Conv2d")
model = alexnet.AlexNet()#pretrained = False)

#Modifying Head - classifier
model.classifier = nn.Sequential(
    nn.Dropout(0.5, inplace = False),
    nn.Linear(9216, 2048, bias = True),
    nn.ReLU(inplace = True),
    nn.Dropout(0.4),
    nn.Linear(2048, 1024, bias = True),
    nn.ReLU(inplace = True),
    nn.Dropout(0.4),
    nn.Linear(1024, 1, bias = True),
    nn.Sigmoid()
)


# %% [code] {"execution":{"iopub.status.busy":"2021-07-08T10:47:50.424995Z","iopub.execute_input":"2021-07-08T10:47:50.425347Z","iopub.status.idle":"2021-07-08T10:47:50.463638Z","shell.execute_reply.started":"2021-07-08T10:47:50.425311Z","shell.execute_reply":"2021-07-08T10:47:50.462635Z"}}
### Defining model parameters

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

# Learning Rate Scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.5)

#Loss Function
criterion = nn.BCELoss()

# Logs - Helpful for plotting after training finishes
train_logs = {"loss" : [], "accuracy" : [], "time" : []}
val_logs = {"loss" : [], "accuracy" : [], "time" : []}

# Loading model to device
model.to(device)

# No of epochs
epochs = 50

# %% [code] {"execution":{"iopub.status.busy":"2021-07-08T10:47:50.465535Z","iopub.execute_input":"2021-07-08T10:47:50.4659Z","iopub.status.idle":"2021-07-08T10:50:09.719593Z","shell.execute_reply.started":"2021-07-08T10:47:50.465865Z","shell.execute_reply":"2021-07-08T10:50:09.718047Z"}}
### Training and Validation xD
for epoch in range(epochs):

    ###Training
    loss, acc, _time = train_one_epoch(train_data_loader)

    #Print Epoch Details
    print("\nTraining")
    print("Epoch {}".format(epoch+1))
    print("Loss : {}".format(round(loss, 4)))
    print("Acc : {}".format(round(acc, 4)))
    print("Time : {}".format(round(_time, 4)))

    ###Validation
    loss, acc, _time = val_one_epoch(val_data_loader)

    #Print Epoch Details
    print("\nValidating")
    print("Epoch {}".format(epoch+1))
    print("Loss : {}".format(round(loss, 4)))
    print("Acc : {}".format(round(acc, 4)))
    print("Time : {}".format(round(_time, 4)))
    vals.append(acc)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-08T10:50:09.721114Z","iopub.status.idle":"2021-07-08T10:50:09.722613Z"}}
### Plotting Results

print(vals)
