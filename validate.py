from train import resume_from_checkpoints, get_models
import config
import os 
import PIL.Image as Image
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import numpy as np 




def load_generator(generator): 
        if config.best_weight_g: 
                trained_weights = torch.load(config.best_weight_g)
                model_weights = generator.state_dict()
                model_weights.update({k:v for k,v in trained_weights.items() if k in model_weights.keys()})
                generator.load_state_dict(model_weights, strict=True)
                print(f"Loaded pretrained weights from {config.best_weight_g} \n")


def load_transforms_lr():
        transform = transforms.Compose([
                transforms.PILToTensor(),
                transforms.Resize((config.lr_size, config.lr_size)),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        ])

        return transform

def load_transforms_hr(): 
        transform = transforms.Compose([
                transforms.PILToTensor(),
                transforms.Resize((config.hr_size, config.hr_size)),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        ])

        return transform


def plot_lr_hr(sr, hr): 
    sr = np.transpose(np.squeeze(sr), [1,2,0])
    hr = np.transpose(hr, [1,2,0])
    f, ax = plt.subplots(1, 2)
    ax[0].imshow(sr)
    ax[1].imshow(hr) 
    plt.show()



def main(): 
        # start training regarding the mode 
        
        print("----- Loading models -----")
        generator, _ = get_models()
        generator.eval()
        print("----- Successfuly loaded models ----- \n")

        print("------ Checking for existing checkpoints ------")
        load_generator(generator)
        print("------ Done with checkpoints ------")

        lr_transform = load_transforms_lr() 
        hr_transform = load_transforms_hr()


        for path in os.listdir(config.test_hr_directory): 
                print(f"Opening file : {path} ....")
                
                file_path = os.path.join(config.test_hr_directory, path)
                pil_image = Image.open(file_path) 
                lr = lr_transform(pil_image).to(config.device)
                hr = hr_transform(pil_image)

                with torch.no_grad(): 
                        # predictions
                        sr = generator(torch.unsqueeze(lr, dim=0)).cpu().numpy()
                        plot_lr_hr(sr, hr.cpu().numpy())
                






if __name__ == "__main__": 
    main()