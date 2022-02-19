from models import Generator
from torchinfo import summary
from train import get_models
import config 

def visualize_generator(): 
    generator, discriminator = get_models()
    summary(generator, input_size=(config.batch_size, 3, config.lr_size, config.lr_size))
    summary(discriminator, input_size=(config.batch_size, 3, config.hr_size, config.hr_size))


def main(): 
    visualize_generator() 

if __name__ == "__main__": 
     main()