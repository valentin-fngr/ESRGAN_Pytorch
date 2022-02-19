import os 
import config 
from scripts.build_data import split_inside_fodler
from dataset import CustomDataset



def main(): 
    """
        Entry point
    """
    if config.split_inside: 
        # split the folder into train test val subfolders
        if config.mode == "train_esrgan": 
            if config.split_inside:
               split_inside_fodler(config.main_folder, config.train_split, config.test_split, config.val_split)
            

if __name__ == "__main__": 
    main()