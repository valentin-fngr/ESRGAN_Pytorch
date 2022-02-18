import torch
from datetime import datetime



# Random seed to maintain reproducible results
torch.manual_seed(0)
# device 
device = torch.device("cuda", 0)

# upsampling coefficient
upsample_coefficient = 4

# mode selection : traning esrgan | validation
mode = "train_esrgan"

# experience 
experience_name = "default_experience_esrgan_" + datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

