from email.policy import strict
import os

from torch import device 
import config 
from scripts.build_data import split_inside_fodler
from dataset import CustomDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np 
from models import Generator, Discriminator, ContentLoss
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
    discriminator = Discriminator().to(config.device)

    generator = Generator(config.lr_size).to(config.device)
    return generator, discriminator
    

def get_optimizers(generator, discriminator): 
    if config.train_mode == "psnr_oriented": 
        lr = config.learning_rate_psnr
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
    psnr_criterion = torch.nn.MSELoss().to(config.device) # to compute the psnr  
    return l1_criterion, vgg_criterion, adversarial_criterion, psnr_criterion 
        

def compute_psnr(hr, sr, psnr_criterion): 
    """
        Computer the psnr
    """
    return 10 * (torch.log10(1/psnr_criterion(hr, sr)))


def validate(generator, val_loader, psnr_criterion, epoch, writer): 
    """
        validation based on psnr
    """

    with torch.no_grad(): 
        psnrs = []
        for i, sample in enumerate(val_loader): 
            hr = sample["hr"].to(config.device) 
            lr = sample["lr"].to(config.device)

            # generate super resolution 
            sr = generator(lr)

            # compute psnr
            psnr = compute_psnr(hr, sr, psnr_criterion) 
            psnrs.append(psnr.mean())


        avg_psnr = torch.Tensor(psnrs).mean()
        writer.add_scalar(f"{config.train_mode}/PSNR", avg_psnr, epoch)
    
    return avg_psnr


def get_scheduler(g_optim, d_optim): 
    g_scheduler = torch.optim.lr_scheduler.StepLR(g_optim, config.decay_time, gamma=config.decay_rate, verbose=True)
    d_scheduler = torch.optim.lr_scheduler.StepLR(d_optim, config.decay_time, gamma=config.decay_rate, verbose=True)

    return g_scheduler, d_scheduler


def resume_from_checkpoints(generator, discriminator): 
    if config.resume or config.train_mode == "post_training": 
        # check for best weights in directories 
        if config.best_weight_g: 
            trained_weights = torch.load(config.best_weight_g)
            model_weights = generator.state_dict()
            model_weights.update({k:v for k,v in trained_weights.items() if k in model_weights.keys()})
            generator.load_state_dict(model_weights, strict=True)
            print(f"Loaded pretrained weights from {config.best_weight_g} \n")

        if config.best_weight_d: 
            trained_weights = torch.load(config.best_weight_d)
            model_weights = discriminator.state_dict()
            model_weights.update({k:v for k,v in trained_weights.items() if k in model_weights.keys()})
            discriminator.load_state_dict(model_weights)
            print(f"Loaded pretrained weights from {config.best_weight_d}")
    else: 
        print(f"Training in {config.train_mode} mode from scratch \n")


def train_psnr(generator, g_optim, train_loader, l1_criterion, writer, epoch): 


    for i, samples in enumerate(train_loader): 
        
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
        writer.add_scalar(f"{config.train_mode}/l1_loss_psnr_state", l1_loss, epoch*len(train_loader) + i + 1)
        
        if i % 50 == 0 and i != 0: 
            print(f"EPOCH={epoch} [{i}/{len(train_loader)}]L1 loss in psnr training mode : {l1_loss} ")  


def train_post_psnr(generator, discriminator, g_optim, d_optim, train_loader, l1_criterion, vgg_criterion, adversarial_criterion, writer, epoch): 

    
    for i, sample in enumerate(train_loader): 
        
        hr = sample["hr"].to(config.device) 
        lr = sample["lr"].to(config.device) 

        # generated super resolution 
        sr = generator(lr)

        ###### train discriminator ######


        for param in discriminator.parameters():
            param.requires_grad = True

        d_optim.zero_grad()

        true_label = torch.full(size=(sr.shape[0], 1), fill_value=1.0, device=config.device)
        fake_label = torch.full(size=(sr.shape[0], 1), fill_value=0.0, device=config.device)

        predicted_true = discriminator(hr)
        predicted_fake = discriminator(sr.detach())

        d_loss_true = adversarial_criterion(torch.sigmoid(predicted_true - predicted_fake.mean(dim=0)), true_label)
        d_loss_fake = adversarial_criterion(torch.sigmoid(predicted_fake - predicted_true.mean(dim=0)), fake_label)
        # optimization 
        
        d_loss = d_loss_fake + d_loss_true
        d_loss.backward()
        d_optim.step()


        ###### train generator ######
        for param in discriminator.parameters():
            param.requires_grad = False
        
        g_optim.zero_grad()

        d_out_generated = discriminator(sr)
        d_out_hr = discriminator(hr.detach())
        # mse/vgg loss
        vgg_loss = vgg_criterion(sr, hr.detach())
        # l1 criterion
        l1_loss = config.l1_coefficient * l1_criterion(sr, hr.detach())
        # relativistic loss
        relativistic_loss = config.relativistic_coefficient * adversarial_criterion(torch.sigmoid(d_out_generated - d_out_hr.mean(dim=0)), true_label)
        # complete loss
        g_loss = vgg_loss + l1_loss  + relativistic_loss
        g_loss.backward()
        # optimization 
        g_optim.step()

        # metrics
        d_hr_prob = torch.sigmoid(d_out_hr.mean(dim=0))
        d_sr_prob = torch.sigmoid(d_out_generated.detach().mean(dim=0))


        # writing with tensorboard
        writer.add_scalar(f"{config.train_mode}/D_LOSS", d_loss, epoch*len(train_loader) + i + 1)
        writer.add_scalar(f"{config.train_mode}/G_LOSS", g_loss, epoch*len(train_loader) + i + 1)
        writer.add_scalar(f"{config.train_mode}/l1_loss", l1_loss, epoch*len(train_loader) + i + 1)
        writer.add_scalar(f"{config.train_mode}/vgg_loss", vgg_loss, epoch*len(train_loader) + i + 1)
        writer.add_scalar(f"{config.train_mode}/relativistic_loss", relativistic_loss, epoch*len(train_loader) + i + 1)
        writer.add_scalar(f"{config.train_mode}/D(HR)", d_hr_prob, epoch*len(train_loader) + i + 1)
        writer.add_scalar(f"{config.train_mode}/D(SR)", d_sr_prob, epoch*len(train_loader) + i + 1)

        if i % 50 == 0 and i != 0: 
            print(f"EPOCH={epoch} [{i}/{len(train_loader)}]D_LOSS in {config.train_mode} mode : {d_loss} ")  
            print(f"EPOCH={epoch} [{i}/{len(train_loader)}]D(HR) in {config.train_mode} : {d_hr_prob} ")
            print(f"EPOCH={epoch} [{i}/{len(train_loader)}]D(SR) in {config.train_mode} : {d_sr_prob} ")    
            print(f"EPOCH={epoch} [{i}/{len(train_loader)}]G_LOSS in {config.train_mode} : {g_loss} ")  


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
        generator, discriminator = get_models()
        print("----- Successfuly loaded models ----- \n")
        
        print("----- Loading losses -----")
        l1_criterion, vgg_criterion, adversarial_criterion, psnr_criterion = define_losses()
        print("----- Successfuly loaded all losses -----")


        print("----- Loading optimizers -----")
        g_optim, d_optim = get_optimizers(generator, discriminator) 
        
        g_scheduler, d_scheduler = get_scheduler(g_optim, d_optim)
        
        print("----- Successfuly loaded all optimizers ----- \n")


        print("----- Initiliazing Tensorboard writer -----")
        writer = SummaryWriter(log_dir=f"runs/{config.experience_name}", comment=config.experience_name)
        print("----- Successfuly initialized a Tensorboard writer ----- \n")
        

        print("------ Checking for existing checkpoints ------")
        resume_from_checkpoints(generator, discriminator)
        print("------ Done with checkpoints ------")


        generator.train()
        discriminator.train()
        best_psnr = 0.0

        for epoch in range(config.epochs): 
            # iteration 
            if config.train_mode == "psnr_oriented": 
                train_psnr(generator, g_optim, train_loader, l1_criterion, writer, epoch) 
            elif config.train_mode == "post_training": 
                train_post_psnr(generator, discriminator, g_optim, d_optim, train_loader, l1_criterion, vgg_criterion, adversarial_criterion, writer, epoch)
            
        
            print("----- Validation step -----")
            psnr = validate(generator, val_loader, psnr_criterion, epoch, writer)
            print(f"----- Validation score on PSNR : {psnr}")
            
            if psnr >= best_psnr: 
                print(f"----- Saving new best weights for epoch {epoch} -----")
                best_psnr = psnr
                torch.save(generator.state_dict(), os.path.join(config.checkpoints_best_g, f"best_weight_gen_{config.train_mode}.pth"))
                if config.train_mode == "post_training":
                    torch.save(discriminator.state_dict(), os.path.join(config.checkpoints_best_d, f"best_weight_dis_{config.train_mode}.pth"))

            torch.save(generator.state_dict(), os.path.join(config.checkpoints_epoch_g, f"g_epoch_{config.train_mode}={epoch+1}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(config.checkpoints_epoch_d, f"d_epoch={epoch+1}_{config.train_mode}.pth"))

            g_scheduler.step() 

            

        

if __name__ == "__main__": 
    main()