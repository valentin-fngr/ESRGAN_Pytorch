import os 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
from PIL import Image


class CustomDataset(Dataset): 

    def __init__(self, folder, upscale_factor, hr_size): 
        self.folder = folder 
        self.files = sorted(glob.glob(folder + "/*.*"))
        self.upscale_factor = upscale_factor
        self.hr_size = hr_size

        self.hr_transforms = transforms.Compose([
            transforms.Resize(hr_size), 
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        ])
        self.lr_transforms = transforms.Compose([
            transforms.Resize(hr_size // upscale_factor), 
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        ])


    def __len__(self): 
        return len(self.files)

    def __getitem__(self, idx): 
        img_name = self.files[idx] 
        image_obj = Image.open(img_name)

        return {
            "hr" : self.hr_transforms(image_obj), 
            "lr" : self.lr_transforms(image_obj)
        } 
