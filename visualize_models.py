from models import Generator, ContentLoss
from torchinfo import summary
from train import get_models
import config 


def visualize_generator(): 
    generator, rela_discriminator = get_models()
    summary(generator, input_size=(config.batch_size, 3, config.lr_size, config.lr_size))
    summary(rela_discriminator.discriminator, input_size=(config.batch_size, 3, config.hr_size, config.hr_size))


def visualize_vgg19(): 
    vgg19 = ContentLoss().vgg19
    summary(vgg19, input_size=(config.batch_size, 3, config.hr_size, config.hr_size))
    

def main(): 
    visualize_generator() 
    visualize_vgg19()

if __name__ == "__main__": 
     main()