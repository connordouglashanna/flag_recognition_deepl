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

src = "C:/Users/condo/OneDrive/Documents/Engineers_for_Ukraine/flag_recognition_deepl"

## now we use the os package to create a list of files in the directory and rename
# os.listdir() finds the files
# os.rename() renames them
country = ["Russian", "Ukrainian", "Soviet"]
last_folder = ["Russian_Flag", "Ukrainian_Flag", "Soviet_Flag"]

# defining a dictionary using these lists
data = {'country': country,
        'directory': last_folder}

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
# https://towardsdatascience.com/building-efficient-custom-datasets-in-pytorch-2563b946fd9f

data_root = "C:/Users/condo/OneDrive/Documents/Engineers_for_Ukraine/flag_recognition_deepl/Flags/"

class FlagsDataset(Dataset):
    
    def __init__(self, data_root):
        self.samples = []
        
        # fetching category folders/names
        for country in os.listdir(data_root):
            country_folder = os.path.join(data_root, country)
            
            # testing
            print(country_folder)
            
            # fetching observation filepaths
            for flag in os.listdir(country_folder):
                flag_filepath = os.path.join(country_folder, flag).replace("\\","/")
                
                # testing
                print(flag_filepath)
                
                # iterating 
                with open(flag_filepath, 'r') as flag_file:
                    for tensor in # insert tensorizing code here: 
                        # populating each sample with obs index, filepath, and category
                        self.samples.append((country, obs, obs_filepath))
    ## potential other steps:
        # unifying resolution of images
        # converting images to tensors
        # one hot encoding country
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
    
# inspecting the dataset observations
if __name__ == '__main__':
    dataset = FlagsDataset(data_root)
    print(len(dataset))
    print(dataset[100])