import os

from torch import device 
import config 
from scripts.build_data import split_inside_fodler
from dataset import CustomDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np 
from models import Generator, RelativisticDiscriminator, Discriminator, ContentLoss
from torchinfo import summary
import torch.optim as optim
import torch.nn as nn



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


def get_models(generator_weights=None, discriminator_weights=None): 
    """
        TODO : add saved weight when resume = True
    """
    # discriminator backbone
    discriminator = Discriminator()

    generator = Generator(config.lr_size).to(config.device)
    relativistic_discriminator = RelativisticDiscriminator(discriminator).to(config.device)
    return generator, discriminator, relativistic_discriminator
    

def get_optimizers(generator, discriminator): 
    generator_optim = optim.Adam(generator.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2))
    discriminator_optim = optim.Adam(discriminator.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2))

    return generator_optim, discriminator_optim 


def plot_image(img_array): 
    img = img_array.detach().cpu().numpy()
    img = np.transpose(img, [1,2,0])
    plt.imshow(img) 
    plt.show()



def define_losses(training_mode): 
    """
        Return losses to train the generator based on the training mode
    """
    # evaluates difference between the genereated sr and hr
    l1_criterion = nn.L1Loss().to(config.device)
    vgg_criterion = ContentLoss().to(config.device)
    adversarial_criterion = nn.BCEWithLogitsLoss().to(config.device)
    return l1_criterion, vgg_criterion
        


def main(): 
    """
        Entry point
    """
    if config.mode == "train_esrgan": 
        if config.split_inside:
            split_inside_fodler(config.main_folder, config.train_split, config.test_split, config.val_split)

        train_loader, val_loader = load_dataset()



if __name__ == "__main__": 
    main()