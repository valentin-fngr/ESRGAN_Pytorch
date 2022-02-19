import torch
from datetime import datetime
import os


# Random seed to maintain reproducible results
torch.manual_seed(0)
# device 
device = torch.device("cuda", 0)

# upsampling coefficient
upsample_coefficient = 4

# mode selection : traning esrgan | validation
mode = "train_esrgan"

# high resolution size 
hr_size = 128

# experience 
experience_name = "default_experience_esrgan_" + datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

# dataset path 
data_dir = os.path.join(os.getcwd(), "data")
print(data_dir)


if mode == "train_esrgan": 

    epochs = 40
    print_frequency = 500
    resume = False 
    
    weight_path = ""
    start_epoch = 0

    # use split_inside = True if you have only one folder containing all images, 
    # so the data can be splitted inside this folder
    split_inside = True
    # use the following splits if you do not have  train test split folders and you wish to divide your main image folder 
    train_split = 0.7 
    test_split = 0.1 
    val_split = 0.2

    # your image folder before train test split folders
    main_folder = os.path.join(data_dir, "faces/img")

    training_data = os.path.join(data_dir, "faces/img/train")
    validation_data = os.path.join(data_dir, "faces/img/val")
    test_data = os.path.join(data_dir, "faces/img/test")
