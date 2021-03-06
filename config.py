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
experience_name = "train_post_psnr" + datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

# dataset path 
data_dir = os.path.join(os.getcwd(), "data")

# leaky relu slope 
lrelu_slope = 0.2

checkpoints_dir = os.path.join(os.getcwd(), "checkpoints")
checkpoints_epoch_g = os.path.join(checkpoints_dir, "on_epoch/generator")
checkpoints_epoch_d = os.path.join(checkpoints_dir, "on_epoch/discriminator")
checkpoints_best_g = os.path.join(checkpoints_dir, "best/generator")
checkpoints_best_d = os.path.join(checkpoints_dir, "best/discriminator")

best_weight_g = "checkpoints/best/generator/best_weight_gen_psnr_oriented.pth"
best_weight_d = ""

# residual scaling
residual_scaling = 0.2

if mode == "train_esrgan": 
    epochs = 40

    # use psnr_oriented if it's the first time you are training the model 
    # use post_training to initialize the model with parameters from psnr_oriented training (generator)
    train_mode = "psnr_oriented"
    learning_rate_psnr = 2*1e-4
    learning_rate_post = 1e-4

    decay_time = epochs // 2
    decay_rate = 2


    # during post_training mode 

    beta1 = 0.9
    beta2 = 0.999
    print_frequency = 500
    resume = False     
    start_epoch = 0
    
    # use split_inside = True if you have only one folder containing all images, 
    # so the data can be splitted inside this folder
    split_inside = False
    # use the following splits if you do not have  train test split folders and you wish to divide your main image folder 
    train_split = 0.7 
    test_split = 0.1 
    val_split = 0.2

    # loss function coefficients
    l1_coefficient = 5*1e-3
    relativistic_coefficient = 1*1e-2 
    adversarial_coefficient = 1e-3

    # your image folder before train test split folders
    main_folder = os.path.join(data_dir, "faces/img")

    training_data = os.path.join(data_dir, "faces/img/train")
    validation_data = os.path.join(data_dir, "faces/img/val")
    test_data = os.path.join(data_dir, "faces/img/test")

elif mode == "validation": 
    
    test_hr_directory = os.path.join(os.getcwd(), "data/Set5/original")
    test_result_directory = os.path.join(os.getcwd(), "results/test")
    