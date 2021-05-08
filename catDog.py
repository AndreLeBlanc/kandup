import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from sklearn.model_selection import train_test_split
import glob
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import os
import zipfile
import random
from sys import argv

#my files
import cnnGabZern
import dataset

lr = 0.001 # learning_rate
batch_size = 96 # we will use mini-batch method
epochs = 10 # How much to train a model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1234)
if device =='cuda':
    torch.cuda.manual_seed_all(1234)

os.listdir('input/dogs-vs-cats-redux-kernels-edition')

os.makedirs('data', exist_ok=True)

base_dir = 'input/dogs-vs-cats-redux-kernels-edition'
train_dir = 'data/train'
test_dir = 'data/test'

#data Augumentation
train_transforms =  transforms.Compose([
        transforms.Resize((220, 220)),
        transforms.RandomResizedCrop(220),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

val_transforms = transforms.Compose([
        transforms.Resize((220, 220)),
        transforms.RandomResizedCrop(220),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

test_transforms = transforms.Compose([
    transforms.Resize((220, 220)),
     transforms.RandomResizedCrop(220),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
    ])

def unZip():
    if os.path.exists("data/train") == False:
         with zipfile.ZipFile(os.path.join(base_dir, 'train.zip')) as train_zip:
            train_zip.extractall('data')

         with zipfile.ZipFile(os.path.join(base_dir, 'test.zip')) as test_zip:
            test_zip.extractall('data')

def createModel(train_data, train_loader, val_data, val_loader, whichNet, kernelSize):
    print(len(train_data), len(train_loader))
    print(len(val_data), len(val_loader))
    print(train_data[0][0].shape)
    if whichNet.upper() == "CNNC":
        model = cnnGabZern.oneConv(kernelSize, "Conv2d").to(device)
    elif whichNet.upper() == "CNN3C":
        model = cnnGabZern.threeConv(kernelSize, "Conv2d").to(device)
    elif whichNet.upper() == "GABORC":
        model = cnnGabZern.oneConv(kernelSize, "Gabor").to(device)
    elif whichNet.upper() == "GABOR3C":
        model = cnnGabZern.threeConv(kernelSize, "Gabor").to(device)
    elif whichNet.upper() == "ZERNC":
        model = cnnGabZern.oneConv(kernelSize, "Zern").to(device)
    elif whichNet.upper() == "ZERN3C":
        model = cnnGabZern.oneConv(kernelSize, "Zern").to(device)
    else:
        print("no model named ", whichNet)
        exit()
    model.train()
    print(model)
    return model

def runEpoch(model, train_loader, optimizer, criterion):
    epoch_loss = 0
    epoch_accuracy = 0

    for data, label in train_loader:
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = ((output.argmax(dim=1) == label).float().mean())
        epoch_accuracy += acc/len(train_loader)
        epoch_loss += loss/len(train_loader)
    return epoch_accuracy, epoch_loss

def validateEpoch(model, val_loader, criterion, epoch):
    with torch.no_grad():
        epoch_val_accuracy=0
        epoch_val_loss =0
        for data, label in val_loader:
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output,label)

            acc = ((val_output.argmax(dim=1) == label).float().mean())
            epoch_val_accuracy += acc/ len(val_loader)
            epoch_val_loss += val_loss/ len(val_loader)

        print('Epoch : {}, val_accuracy : {}, val_loss : {}'
              .format(epoch+1, epoch_val_accuracy, epoch_val_loss))

def runAllEpochs(model, train_loader, val_loader):
    optimizer = optim.Adam(params = model.parameters(),lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        epoch_accuracy, epoch_loss = runEpoch(model, train_loader, optimizer, criterion)
        print('Epoch : {}, train accuracy : {}, train loss : {}'
              .format(epoch+1, epoch_accuracy,epoch_loss))
        validateEpoch(model, val_loader, criterion, epoch)

def dogProb(model, test_loader):
    dog_probs = []
    model.eval()
    with torch.no_grad():
        for data, fileid in test_loader:
            data = data.to(device)
            preds = model(data)
            preds_list = F.softmax(preds, dim=1)[:, 1].tolist()
            dog_probs += list(zip(list(fileid), preds_list))

    idx = list(map(lambda x: x[0],dog_probs))
    prob = list(map(lambda x: x[1],dog_probs))
    submission = pd.DataFrame({'id':idx,'label':prob})
    id_list = []
    class_ = {0: 'cat', 1: 'dog'}
    fig, axes = plt.subplots(2, 5, figsize=(20, 12), facecolor='w')

    for ax in axes.ravel():
        i = random.choice(submission['id'].values)
        label = submission.loc[submission['id'] == i, 'label'].values[0]
        if label > 0.5:
            label = 1
        else:
            label = 0

    img_path = os.path.join(test_dir, '{}.jpg'.format(i))
    img = Image.open(img_path)

    ax.set_title(class_[label])
    ax.imshow(img)

def main():
    unZip()
    train_list = glob.glob(os.path.join(train_dir,'*.jpg'))
    test_list = glob.glob(os.path.join(test_dir, '*.jpg'))
    train_list, val_list = train_test_split(train_list, test_size=0.2)
    train_data = dataset.dataset(train_list, transform=train_transforms)
    test_data = dataset.dataset(test_list, transform=test_transforms)
    val_data = dataset.dataset(val_list, transform=test_transforms)
    train_loader = torch.utils.data.DataLoader(
        dataset = train_data, batch_size=batch_size, shuffle=True )
    test_loader = torch.utils.data.DataLoader(
        dataset = test_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        dataset = val_data, batch_size=batch_size, shuffle=True)

    #create model
    model = createModel(train_data, train_loader, val_data, val_loader, argv[1], int(argv[2]))

    #train model
    runAllEpochs(model, train_loader, val_loader)

    #result
#    dogProb(model, test_loader)

if __name__ == '__main__':
    main()
