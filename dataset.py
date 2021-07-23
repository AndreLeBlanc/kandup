import torch
import torch.nn.functional as F
from PIL import Image
from tifffile import imread, imwrite
from torchvision.transforms import ToTensor
import cv2

class dataset(torch.utils.data.Dataset):
    def __init__(self,file_list,transform=None):
        self.file_list = file_list
        self.transform = transform


    #dataset length
    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    #load one of the images
    def __getitem__(self,idx):
        img_path = self.file_list[idx]
        if img_path.split('/')[-1].split('.')[2] == "tif":
            img = imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_transformed = ToTensor()(img)
        else:
            img = Image.open(img_path)
            img_transformed = self.transform(img)
        label = img_path.split('/')[-1].split('.')[0]
        if label == 'dog':
            label=1
        elif label == 'cat':
            label=0

        return img_transformed,label
