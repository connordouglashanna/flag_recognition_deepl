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

src = "C:/Users/condo/OneDrive/Documents/Engineers_for_Ukraine/flag_recognition_deepl"

# now we use the os package to create a list of files in the directory and rename
# os.listdir() finds the files
# os.rename() renames them
country = ["Russia", "Ukraine", "Soviet"]
last_folder = ["Russian_Flag", "Ukrainian_Flag", "Soviet_Flag"]

# defining a dictionary using these lists
data = {'country': country,
        'directory': last_folder}

df = pd.DataFrame(data, columns=["country", "directory"])

for i in range(len(df)):
    
    # specifying the folder the country flags are located in
    folder = "C:/Users/condo/OneDrive/Documents/Engineers_for_Ukraine/flag_recognition_deepl/Flags/" + df.loc[i, "directory"]
    # specifying the name for use in the renaming loop
    country = country
    
    # now iterating across each file in the directory
    for count, filename in enumerate(os.listdir(folder)):
        dst = f"{country}_{str(count)}.jpg"
        src = f"{folder}/{filename}"
        dst = f"{folder}/{dst}"
    
        # renaming based on list
        os.rename(src, dst)