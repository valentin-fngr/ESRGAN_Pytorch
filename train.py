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
from torch.utils.tensorboard import SummaryWriter
import torch


def plot_image(img_array): 
    img = img_array.detach().cpu().numpy()
    img = np.transpose(img, [1,2,0])
    plt.imshow(img) 
    plt.show()



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
    if config.train_mode == "pnsr_oriented": 
        lr = config.learning_rate_pnsr
    elif config.train_mode == "post_training": 
        lr = config.learning_rate_post

    generator_optim = optim.Adam(generator.parameters(), lr=lr, betas=(config.beta1, config.beta2))
    discriminator_optim = optim.Adam(discriminator.parameters(), lr=lr, betas=(config.beta1, config.beta2))

    return generator_optim, discriminator_optim 


def define_losses(): 
    """
        Return losses to train the generator based on the training mode
    """
    # evaluates difference between the genereated sr and hr
    l1_criterion = nn.L1Loss().to(config.device)
    vgg_criterion = ContentLoss().to(config.device)
    adversarial_criterion = nn.BCEWithLogitsLoss().to(config.device) # with logit !
    pnsr_criterion = torch.nn.MSELoss().to(config.device) # to compute the pnsr  
    return l1_criterion, vgg_criterion, adversarial_criterion, pnsr_criterion 
        

def compute_psnr(hr, sr, pnsr_criterion): 
    """
        Computer the pnsr
    """
    return 10 * (torch.log10(1/pnsr_criterion(hr, sr)))


def validate(generator, val_dataloader, pnsr_criterion, epoch, writer): 
    """
        validation based on pnsr
    """

    with torch.no_grad(): 
        psnrs = []
        for i, sample in enumerate(val_dataloader): 
            hr = sample["hr"].to(config.device) 
            lr = sample["lr"].to(config.device)

            # generate super resolution 
            sr = generator(lr)

            # compute psnr
            psnr = compute_psnr(hr, sr, pnsr_criterion) 
            psnrs.append(psnr.mean())


        avg_psnr = torch.Tensor(psnrs).mean()
        writer.add_scalar("Validation/PSNR", avg_psnr, i+1)
    
    return avg_psnr





def train_psnr(generator, g_optim, train_dataloader, l1_criterion, writer, epoch): 


    for i, samples in enumerate(train_dataloader): 
        
        # load data 
        hr = samples["hr"].to(config.device) 
        lr = samples["lr"].to(config.device)

        # generate super resolution 
        sr = generator(lr)

        # compute l1 loss
        l1_loss = l1_criterion(hr, sr)
        generator.zero_grad()
        l1_loss.backward() 
        g_optim.step()

        # writing with tensorboard
        writer.add_scalar("Metric/l1_loss_pnsr_state", l1_loss, epoch*len(train_dataloader) + i + 1)
        
        if i % 50 == 0 and i != 0: 
            print(f"EPOCH={epoch} [{i}/{len(train_dataloader)}]L1 loss in pnsr training mode : {l1_loss} ")  


def main(): 
    """
        Entry point
    """
    if config.mode == "train_esrgan": 
        if config.split_inside:
            split_inside_fodler(config.main_folder, config.train_split, config.test_split, config.val_split)

        print("----- Loading the training and validation data -----")
        train_loader, val_loader = load_dataset()
        
        print("----- Successfuly loaded training and validation data ----- \n")
        # start training regarding the mode 
        
        print("----- Loading models -----")
        generator, discriminator, relativistic_discriminator = get_models()
        print("----- Successfuly loaded models ----- \n")
        
        print("----- Loading losses -----")
        l1_criterion, vgg_criterion, adversarial_criterion, pnsr_criterion = define_losses()
        print("----- Successfuly loaded all losses -----")


        print("----- Loading optimizers -----")
        if config.train_mode == "pnsr_oriented": 
            g_optim, d_optim = get_optimizers(generator, discriminator) 
        elif config.train_mode == "post_training": 
            g_optim, d_optim = get_optimizers(generator, relativistic_discriminator)       
        print("----- Successfuly loaded all optimizers ----- \n")


        print("----- Initiliazing Tensorboard writer -----")
        writer = SummaryWriter(log_dir=f"runs/{config.experience_name}", comment=config.experience_name)
        print("----- Successfuly initialized a Tensorboard writer ----- \n")
        
        generator.train()
        best_psnr = 0.0

        for epoch in range(config.epochs): 
            # iteration 
            if config.train_mode == "pnsr_oriented": 
                train_psnr(generator, g_optim, train_loader, l1_criterion, writer, epoch) 
            elif config.train_mode == "post_training": 
                pass
            
            
            print("----- Validation step -----")
            psnr = validate(generator, val_loader, pnsr_criterion, epoch, writer)
            print(f"----- Validation score on PSNR : {psnr}")
            
            if psnr >= best_psnr: 
                print(f"----- Saving new best weights for epoch {epoch} -----")
                best_psnr = psnr
                torch.save(generator.state_dict(), os.path.join(config.checkpoints_best, f"best_weight_gen_{epoch}.pth"))

            torch.save(generator.state_dict(), os.path.join(config.checkpoints_epoch, f"g_epoch={epoch+1}.pth"))

            

            


if __name__ == "__main__": 
    main()