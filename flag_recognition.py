# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 15:21:10 2022

@author: condo
"""

# baby's first independent deepl project

## protip from Reid: 
    # pytorch dataset class

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
        except PermissionError:
            print(filename + " is currently open in the Dataset object and cannot be renamed.")
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
                
                # populating each sample with obs index, filepath, tensor, and category
                self.samples.append([country, obs_id, flag_filepath, flag_img])

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # applying our transform function, if specified
        if self.transform:
            # iterating over the list of samples
            for i in self.samples:
                # applying the transform to the tensor element in the obs-level sublist
                i[3] = self.transform(i[3])
                
        # fetching the transformed samples
        return self.samples[idx]
    
        
# defining file repo
data_root = "C:/Users/condo/OneDrive/Documents/Engineers_for_Ukraine/flag_recognition_deepl/Flags/"

# defining test transform
transform_test = transforms.Compose([
    transforms.PILToTensor()
    ])

            # crop to specified aspect ratio
            
            # shrink to appropriate size for nn
            # use pillow resize() method?
            
            # normalize images
            
            # dimensionality reduction?

# defining main function
if __name__ == '__main__':
    dataset = FlagsDataset(data_root, transform_test)
    print(len(dataset))
    print(dataset[100])
    
