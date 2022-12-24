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

src = "C:/Users/condo/OneDrive/Documents/Engineers_for_Ukraine/flag_recognition_deepl"

# now we use the os package to create a list of files in the directory and rename
# os.listdir() finds the files
# os.rename() renames them

# writing the function
def mass_rename():
    
    # iterating across each file in the directory 
    # defines new name and location
    folder = "C:/Users/condo/OneDrive/Documents/Engineers_for_Ukraine/flag_recognition_deepl/Flags/Russian_Flag"
    for count, filename in enumerate(os.listdir(folder)):
        dst = f"Russian_{str(count)}.jpg"
        src = f"{folder}/{filename}"
        dst = f"{folder}/{dst}"
    
        # renaming based on list
        os.rename(src, dst)

# Driver code
if __name__ == '__mass_rename__':
    
    # Calling main() function
    mass_rename()