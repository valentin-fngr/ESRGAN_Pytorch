import torch
from datetime import datetime
import os


# Random seed to maintain reproducible results
torch.manual_seed(0)
# device 
device = torch.device("cuda", 0)


# mode selection : traning esrgan | validation
mode = "train_esrgan"

# high resolution size 
hr_size = 128
# upsampling coefficient
upsample_coefficient = 4

lr_size = hr_size // upsample_coefficient


# batch size 
batch_size = 16

# experience 
experience_name = "default_experience_esrgan_" + datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

# dataset path 
data_dir = os.path.join(os.getcwd(), "data")

# leaky relu slope 
lrelu_slope = 0.2

# residual scaling
residual_scaling = 0.2

if mode == "train_esrgan": 

    # use psnr_oriented if it's the first time you are training the model 
    # use post_training to initialize the model with parameters from psnr_oriented training (generator)
    train_mode = "pnsr_oriented"
    learning_rate_pnsr = 2*10e-4
    learning_rate_post = 10e-4
    beta1 = 0.9
    beta2 = 0.999
    epochs = 40
    print_frequency = 500
    resume = False 
    
    weight_path = ""
    start_epoch = 0
    # use split_inside = True if you have only one folder containing all images, 
    # so the data can be splitted inside this folder
    split_inside = False
    # use the following splits if you do not have  train test split folders and you wish to divide your main image folder 
    train_split = 0.7 
    test_split = 0.1 
    val_split = 0.2

    # loss function coefficients
    l1_coefficient = 5*10e-3
    relativistic_coefficient = 1*10e-2 

    # your image folder before train test split folders
    main_folder = os.path.join(data_dir, "faces/img")

    training_data = os.path.join(data_dir, "faces/img/train")
    validation_data = os.path.join(data_dir, "faces/img/val")
    test_data = os.path.join(data_dir, "faces/img/test")
