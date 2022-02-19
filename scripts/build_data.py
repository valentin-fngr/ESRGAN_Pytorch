import os
import torch
import numpy as np
import shutil

def split_inside_fodler(folder_path, train_split=0.7, test_split=0.1, val_split=0.2):
    """
        Splits a folder into 3 dataset
    """
    if train_split + test_split + val_split != 1: 
        raise ValueError("Please, make sure your splits size sum up to 1")
    elif train_split < 0 or test_split < 0 or val_split < 0 or train_split > 1 or test_split > 1 or val_split > 1: 
        raise ValueError("Please, make sure to choose splits between 0 and 1")

    number_samples = len(os.listdir(folder_path))
    size_train = int(number_samples * train_split)
    size_test = int(number_samples * test_split) 
    size_val = int(number_samples * val_split)

    print(f"Detected {number_samples} files in data folder")

    images = np.array([os.path.join(folder_path, path) for path in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, path))])

    np.random.shuffle(images)

    
    # create subfolders
    splits = ("train", "test", "val")
    for split in splits: 
        # if one of the folder exists, we abort the operation
        if os.path.exists(os.path.join(folder_path, split)): 
            raise ValueError(f"Folder {splits} already exists. If you wish to split again, remove this folder")
        else:
            os.makedirs(os.path.join(folder_path, split))
            print(f"Created folder {split} inside {folder_path}")
     
    
    if train_split: 
        train_images = images[:size_train]
        for file_path in train_images: 
            shutil.move(file_path, os.path.join(folder_path, "train"))
        print(f"Train folder contains : {len(os.listdir(os.path.join(folder_path, 'train')))}, Expected : {size_train}")
    if test_split: 
        test_images = images[size_train:size_train + size_test]
        for file_path in test_images: 
            shutil.move(file_path, os.path.join(folder_path, "test"))
        print(f"Test folder contains : {len(os.listdir(os.path.join(folder_path, 'test')))}, Expected : {size_test}")

    if val_split: 
        val_images = images[size_train + size_test : size_train + size_test + size_val]
        for file_path in val_images: 
            shutil.move(file_path, os.path.join(folder_path, "val"))
        print(f"Val folder contains : {len(os.listdir(os.path.join(folder_path, 'val')))}, Expected : {size_val}")


    print("Done splitting : Please, make sure you don't have any images left in your main folder \n")
    return 
    

    



