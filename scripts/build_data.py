import os  
import torch

def train_test_loader(full_dataset, test_split=0.15, validation_split=0.2):
    if test_split + validation_split >= 1: 
        raise ValueError("test_split and validation_split can't sum to 1") 
    
    train_split = 1 - (test_split + validation_split)
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_split, validation_split, test_split])
    
    return train_dataset





