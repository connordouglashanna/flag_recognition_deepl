# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 15:21:10 2022

@author: condo
"""

# baby's first independent deepl project

# for image processing I know the following three libraries are related but
# I'm not sure which does what... 

## protip from Reid: 
    # pytorch dataset class

# pillow
# skimage
# numpy.ndarrays

## steps for each:

# crop to specified aspect ratio

# shrink to appropriate size for nn
# use pillow resize() method

# normalize images

# dimensionality reduction?


import torch
import os
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

src = "C:/Users/condo/OneDrive/Documents/Engineers_for_Ukraine/flag_recognition_deepl"

## now we use the os package to create a list of files in the directory and rename
# os.listdir() finds the files
# os.rename() renames them

# defining lists for dictionary
country = ["Russian", "Ukrainian", "Soviet"]
last_folder = ["Russian_Flag", "Ukrainian_Flag", "Soviet_Flag"]

# defining a dictionary using lists
data = {'country': country,
        'directory': last_folder}

# converting dictionary into df
df = pd.DataFrame(data, columns=["country", "directory"])

for i in range(len(df)):
    
    print(df.loc[i, "directory"])
    # specifying the folder the country flags are located in
    folder = "C:/Users/condo/OneDrive/Documents/Engineers_for_Ukraine/flag_recognition_deepl/Flags/" + df.loc[i, "directory"]
    # specifying the name for use in the renaming loop
    country = df.loc[i, "country"]
    
    # now iterating across each file in the directory
    for count, filename in enumerate(os.listdir(folder)):
        dst = f"{country}_{str(count)}.jpg"
        src = f"{folder}/{filename}"
        dst = f"{folder}/{dst}"
    
        # renaming based on list
        # using a try-else block to close out errors
        try:
            os.rename(src, dst)
        except FileExistsError:
            print(filename + " already exists and cannot be renamed.") 
            # now our flag files all have appropriate names without conflicts

## now to read the files 
## https://towardsdatascience.com/building-efficient-custom-datasets-in-pytorch-2563b946fd9f

## note: good practice is to define transform as a variable so it can be changed for test/train
## https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

# defining our dataset class
class FlagsDataset(Dataset):
    
    def __init__(self, data_root, transform=None):
        # defining an empty array to store our data
        self.samples = []
        # defining our transform function
        self.transform = transform
        
        # fetching category folders/names
        for country in os.listdir(data_root):
            country_folder = os.path.join(data_root, country)
            
            # testing
            print(country_folder)
            
            # fetching observation filepaths
            for obs_id in os.listdir(country_folder):
                flag_filepath = os.path.join(country_folder, obs_id).replace("\\","/")
                
                # testing
                print(flag_filepath)
                
                # opening the images using PIL
                flag_img = Image.open(flag_filepath, 'r')
                
                # code removed for now but left for later
                ### designating the opened image files
                ###with Image.open(flag_filepath, 'r') as flag_file:
                ###    for  ___ in flag_file,
                
                # populating each sample with obs index, filepath, tensor, and category
                self.samples.append((country, obs_id, flag_filepath, flag_img))
                
                
                
    ## potential other steps:
        # unifying resolution of images
        # converting images to tensors
        # one hot encoding country
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
    
        # applying our transform function, if specified
        if self.transform:
            sample = self.transform(sample)
        
# defining file repo
data_root = "C:/Users/condo/OneDrive/Documents/Engineers_for_Ukraine/flag_recognition_deepl/Flags/"

# defining transform
transform_train = transforms.Compose([
    transforms.CenterCrop(10),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
    ])

# inspecting the dataset observations
if __name__ == '__main__':
    dataset = FlagsDataset(data_root, transform_train
    print(len(dataset))
    print(dataset[100])