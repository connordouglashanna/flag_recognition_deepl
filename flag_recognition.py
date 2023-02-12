# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 15:21:10 2022

@author: condo
"""

# baby's first independent deepl project

## protip from Reid: 
    # pytorch dataset class
import csv
import torch
import os
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder

#%% File Organization

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
            
#%% Annotation .csv generation

# note that this function comes from the csv module which i will need to install 
def build_csv(directory_string, output_csv_name):
    """Builds a csv file for pytorch training from a directory of folders of images.
    Install csv module if not already installed.
    Args: 
    directory_string: string of directory path, e.g. r'.\data\train'
    output_csv_name: string of output csv file name, e.g. 'train.csv'
    Returns:
    csv file with file names, file paths, class names and class indices
    """
    directory = directory_string
    class_lst = os.listdir(directory)
    class_lst.sort() # important to avoid the random order assigned by os.listdir() giving variable index values
    with open(output_csv_name, 'w', newline = '') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        # now we add in the information previously contained in the directory crawler from the __init__ def
        writer.writerow(['obs_id', 'file_path', 'class_name', 'class_idx'])
        for class_name in class_lst:
            # concats the country name to the og filepath to give the directory with the country folder open
            class_path = os.path.join(directory, class_name) 
            file_list = os.listdir(class_path) # GIVE ME ALL YOUR FILEPATHS
            for file_name in file_list:
                # glueing together pieces to give us the full directory of the file
                file_path = os.path.join(directory, class_name, file_name) 
                # adding this information to the csv file
                writer.writerow([file_name, file_path, class_name, class_lst.index(class_name)]) 
    return

# now defining the data root,
data_folder = "C:/Users/condo/OneDrive/Documents/Engineers_for_Ukraine/flag_recognition_deepl/Flags/"

# original author included separate statements for building train and test datasets. 
# how can I make this work with the dataloaders so that I don't wind up predefining a narrow sample?
build_csv(data_folder, 'flags.csv')
flags_df = pd.read_csv('flags.csv')
            

#%% Dataset class definition

## now to read the files 
## https://towardsdatascience.com/building-efficient-custom-datasets-in-pytorch-2563b946fd9f

## note: good practice is to define transform as a variable so it can be changed for test/train
## https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

# defining our dataset class
class FlagsDataset(Dataset):
    
    def __init__(self, data_root=None, transform=None):
        # defining our root directory attribute
        self.data_root = data_root
        # defining an empty array to store our data
        self.samples = []
        # defining our encoder codex for country
        self.country_codec = LabelEncoder()
        # defining our transform function
        self.transform = transform
        # running our _init_dataset() function
        self._init_dataset()

    def __len__(self):
        return len(self.samples)
    
    ### getitem function needs to run the one-hot encoder
    ### consider using pytorch documentation as model
    def __getitem__(self, idx):
        # applying our transform function, if specified
        if self.transform:
            # iterating over the list of samples
            for i in self.samples:
                # applying the transform to the tensor element in the obs-level sublist
                i[3] = self.transform(i[3])
        
        # fetching the transformed samples
        return self.one_hot_sample(country)
    
    # removing the filepath business to a separate function definition
    def _init_dataset(self):
        countries = set()
        
        # fetching category folders/names
        for country in os.listdir(self.data_root):
            country_folder = os.path.join(self.data_root, country)
            
            # adding country categories
            countries.add(country)
            
            # fetching observation filepaths
            for obs_id in os.listdir(country_folder):
                flag_filepath = os.path.join(country_folder, obs_id).replace("\\","/")
                
                # opening the images using PIL
                flag_img = Image.open(flag_filepath, 'r')
                
                # populating each sample with obs index, filepath, tensor, and category
                self.samples.append([country, obs_id, flag_filepath, flag_img])
        
        # adding the countries to the codex
        self.country_codec.fit(list(countries))
        
    # building a one-hot encoder
    def to_one_hot(self, codec, values):
        value_idxs = codec.transform(values)
        return torch.eye(len(codec.classes_))[value_idxs]
    
    # running the one-hot encoder on data
    def one_hot_sample(self, country):
        t_country = self.to_one_hot(self.country_codec, [country])
        return t_country
        
        
    
# defining file repo
data_root = "C:/Users/condo/OneDrive/Documents/Engineers_for_Ukraine/flag_recognition_deepl/Flags/"

# defining test transform
transform_test = transforms.Compose([
    transforms.PILToTensor()
    ])

# testing a few properties using a main function call
if __name__ == '__main__':
    dataset = FlagsDataset(data_root, transform_test)
    # print statement to check observations
    print("This dataset contains a total of " + str(len(dataset)) + " observations.")
    # inspecting an individual observation
    print(dataset[100:105])
    

#%% Test/train data definitions

# building transforms for our dataset:
    
            # crop to specified aspect ratio
            # shrink to appropriate size for nn
            # use pillow resize() method?
            # normalize images
            # dimensionality reduction?
            
# train transform definition
#transform_train = transforms.Compose([
#    transforms.PILToTensor(),
 #   transforms.FiveCrop(),
 #   ])

# test transform definition
#transform_test = transforms.Compose([
 #   transforms.PILToTensor(),
  #  ])

# defining our test/train datasets

            # how to incorporate test/train split?

#%% Dataloading using dataloaders

    
