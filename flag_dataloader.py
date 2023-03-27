"""
Created on Tue Dec 20 15:21:10 2022

@author: condo
"""

## protip from Reid: 
    # pytorch dataset class
import csv
import torch
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from torch import save, load


class FlagsDataset(Dataset):
    
    def __init__(self, transform=None):
        self.transform = transform

        data_folder = "./Flags" 
        categories = ["Russian", "Ukrainian", "Soviet"]

        image_paths = []
        labels = []

        for ii, category in enumerate(categories):
            flag_path = os.path.join(data_folder, f"{category}_Flag")
            for file_name in os.listdir(flag_path):
                if file_name.endswith('.jpg'):
                    image_path = os.path.join(flag_path, file_name)
                    image_paths.append(image_path)
                    labels.append(ii)

        self.image_paths = image_paths
        self.labels = labels


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # fetching our filepath from the annotation .csv
        image_path = self.image_paths[idx] 
        # reading the images using PIL
        image = Image.open(image_path, 'r')
        # label
        label = torch.tensor([self.labels[idx]])
        ## performing our transformations:
        if self.transform:
            image = self.transform(image)
        # returning our sample
        image = image[:, :100, :100]
        return image, label 



# defining test transform
transform_pilot = transforms.Compose([
    transforms.PILToTensor()
    ])

# testing a few properties using a main function call
dataset = FlagsDataset(transform = transform_pilot)

print(f"This dataset contains a total of {len(dataset)} observations.")


# Split the dataset into training and testing sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create dataloaders for the training and testing sets
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)




for ii, train_batch in enumerate(train_dataloader):
    print(ii)
    train_image, train_label  = train_batch
    print(f'train_image.shape {train_image.shape}')
    print(f'train_label.shape {train_label.shape}')



