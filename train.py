import os 
import config 
from scripts.build_data import split_inside_fodler
from dataset import CustomDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np 

def load_dataset(): 

    train_dataset = CustomDataset(config.training_data, config.upsample_coefficient, config.hr_size)
    val_dataset = CustomDataset(config.validation_data, config.upsample_coefficient, config.hr_size)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        drop_last=True
    )

    return train_loader, val_loader


def plot_image(img_array): 
    img = img_array.detach().cpu().numpy()
    img = np.transpose(img, [1,2,0])
    plt.imshow(img) 
    plt.show()


def main(): 
    """
        Entry point
    """
    if config.mode == "train_esrgan": 
        if config.split_inside:
            split_inside_fodler(config.main_folder, config.train_split, config.test_split, config.val_split)

        train_loader, val_loader = load_dataset()
        
        samples = iter(train_loader).next()
        
        hr = samples["hr"]
        lr = samples["lr"]
        plot_image(hr[1])
        plot_image(lr[1])



if __name__ == "__main__": 
    main()