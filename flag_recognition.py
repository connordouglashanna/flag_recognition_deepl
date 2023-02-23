# -*- coding: utf-8 -*-
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
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.autograd import Variable

#%% Mass renaming

# using the os package to create a list of files in the directory and rename
# defining data folder for use throughout
data_folder = "C:\\Users\condo\OneDrive\Documents\Engineers_for_Ukraine\\flag_recognition_deepl\\Flags"

# defining an empty source directory for this operation
src = ""

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
    folder = os.path.join(data_folder, df.loc[i, "directory"])
    # specifying the name for use in the renaming loop
    country = df.loc[i, "country"]
    
    # now iterating across each file in the directory
    for count, filename in enumerate(os.listdir(folder)):
        dst = f"{country}_{str(count)}.jpg"
        src = f"{folder}\\{filename}"
        dst = f"{folder}\\{dst}"
    
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

# note for future reference that this entire thing could have been done with the native ImageFolder function
# https://blog.paperspace.com/dataloaders-abstractions-pytorch/

# note that this function comes from the csv module which i will need to install 
def build_csv(directory_string, output_csv_name):
    """Builds a csv file for pytorch training from a directory of folders of images.
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


# original author included separate statements for building train and test datasets. 
# how can I make this work with the dataloaders so that I don't wind up predefining a narrow sample?

# note: data_folder object from earlier is reused here
### is this a problem? why is the flag file in the repo folder even though the data folder is specified?
build_csv(data_folder, 'flags.csv')
flags_df = pd.read_csv('flags.csv')
            

#%% Dataset class definition

## now to read the files 
## https://towardsdatascience.com/building-efficient-custom-datasets-in-pytorch-2563b946fd9f

## note: good practice is to define transform as a variable so it can be changed for test/train
## https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

# defining our dataset class
class FlagsDataset(Dataset):
    
    def __init__(self, csv_file, data_root=None, transform=None):
        # defining our root directory attribute
        self.data_root = data_root
        # annotation dataframe
        self.annotation_df = pd.read_csv(csv_file)
        # defining our transform function
        self.transform = transform
        ### note that size argument is required here

    def __len__(self):
        return len(self.annotation_df)

    def __getitem__(self, idx):
        # fetching our filepath from the annotation .csv
        image_path = self.annotation_df.iloc[idx, 1]
        # reading the images using PIL
        image = Image.open(image_path, 'r')
        # defining the class name for that image from the annotations
        class_name = self.annotation_df.iloc[idx, 2]
        # defining the class index for that image from the annotations
        class_index = self.annotation_df.iloc[idx, 3]
        ## performing our transformations:
        if self.transform:
            image = self.transform(image)
        # returning our sample
        return image, class_name, class_index


# defining csv location 
csv_file = "C:\\Users\condo\OneDrive\Documents\Engineers_for_Ukraine\\flag_recognition_deepl\\flags.csv"

## note: data_folder object is reused again below

# defining test transform
transform_pilot = transforms.Compose([
    transforms.PILToTensor()
    ])

# testing a few properties using a main function call
if __name__ == '__main__':
    flags_pilot = FlagsDataset(csv_file = csv_file, 
                           data_root = data_folder, 
                           transform = transform_pilot)
    # print statement to check observations
    print("This dataset contains a total of " + str(len(flags_pilot)) + " observations.")
    # inspecting five individual tensorized observations at random
    for i in range(3):
        idx = torch.randint(len(flags_pilot), (1,))
        idx = idx.item()
        print(flags_pilot[idx])
    
#%% Visualization test

# generating an untransformed dataset to preserve our image characteristics
flags_naked = FlagsDataset(csv_file = csv_file, 
                           data_root = data_folder)

# generating our visualizations
# setting our figure size...
plt.figure(figsize = (12, 6))
# looping to get some images
for i in range(10):
    idx = torch.randint(len(flags_naked), (1,))
    idx = idx.item()
    image, class_name, class_index = flags_naked[idx]
    ax = plt.subplot(2, 5, i + 1) 
    ax.title.set_text(class_name + "_" + str(class_index))
    plt.imshow(image)

#%% Test/train split definitions & transforms

### building transforms for our dataset: needs update
### note that updates should come after model is built to aid in tuning
            # desired transforms include: 
                #transforms.RandomHorizontalFlip()
                #transforms.normalize()
                # set size to 100x100
                #transforms.FiveCrop()
                #transforms.Resize()
            
# train transform definition
transform_train = transforms.Compose([
    transforms.PILToTensor(),
    transforms.Resize(200),
    transforms.RandomCrop(150),
    transforms.RandomHorizontalFlip(),
    # normalizing based on arbitrary guidance from StackOverflow
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

# test transform definition
# same as train for now... why do I want this to be the same? Do I?
transform_test = transforms.Compose([
    transforms.PILToTensor(),
    transforms.Resize(200),
    transforms.RandomCrop(150),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

# defining our test/train datasets
### these remain separate just in case I decide to do independent transform configs
# train dataset
flags_train = FlagsDataset(csv_file = csv_file, 
                           data_root = data_folder, 
                           transform = transform_train)
# test dataset
flags_test = FlagsDataset(csv_file = csv_file, 
                          data_root = data_folder, 
                          transform = transform_test)

#%% Visualizing our transformed images

### add visualizations???

#%% Defining our dataloaders

# training dataloader
dataloader_train = torch.utils.data.DataLoader(flags_train,
                                               ### how do I select batch size?
                                               batch_size = 50,
                                               shuffle = True,
                                               num_workers = 0),

# testing dataloader
dataloader_test = torch.utils.data.DataLoader(flags_test, 
                                              batch_size = 50,
                                              shuffle = True,
                                              num_workers = 0),
    
### do I even need to use dataloaders at all?
### note: dataloaders seem to be critical in batching/setting batch size for data. That appears to be the whole purpose.

#%% Model definition 

# selecting a device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
    
### the official PyTorch documentation version:
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

# and the official task-specific PyTorch documentation/explainer:
# https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

# link explaining how/why layers are chosen:
# https://stackoverflow.com/questions/53784998/how-are-the-pytorch-dimensions-for-linear-layers-calculated

# queueing network definition
class Net(nn.Module):
    # initializing neural network
    # this contains all of our neural network layers...
    def __init__(self):
        # the arguments in super() differ between the two tutorials
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # these appear to be the linear processing layers, in second tutorial
        # "applies a linear transformation to an object having specified shape, features(?)
        # these alternate with ReLU feedback functions....
        # "applies the rectified linear unit function element-wise"
        # has args dimensions, das it
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

### section needs heavy revision
class FlagsCNN(nn.Module):
    def __init__(self):
        super(FlagsCNN, self).__init__()
        # these lines define convolutions
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                ### channels???
                in_channels = 1,
                out_channels = 16, 
                kernel_size = 5,
                stride = 1,
                padding = 2,
                ),
            # optimization function, most likely leave this alone
            nn.ReLU(),
            ### do we have a 2d image???
            nn.MaxPool2d(kernel_size = 2),
            )
        # this model, as written, has two convolution layers
        ### how many convolution layers do we want?
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        ### how do we determine the contents of the convolution layers?
        
        # fully connected layer, output 3 classes
        ### what are the other dimensions doing here???
        self.out = nn.Linear(32 * 7 * 7, 3)
        
    def forward(self, x):
        ### what is happening here???
        x = self.conv1(x)
        x = self.conv2(x)
        
        # flattening the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x # now we have x for visualization


#%% Training the model 

