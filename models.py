import torch 
import numpy as np 
import torch.nn as nn 
import torch.nn.functional as F
import config
import torchvision


class RRDB(nn.Module): 
    """
        RRDB block
    """ 
    
    def __init__(self, in_c, k=3): 
        super(RRDB, self).__init__() 
        self.conv1 = nn.Conv2d(in_c, in_c, k, padding=1)
        self.conv2 = nn.Conv2d(in_c, in_c, k, padding=1)
        self.conv3 = nn.Conv2d(in_c, in_c, k, padding=1)
        self.conv4 = nn.Conv2d(in_c, in_c, k, padding=1)
        self.conv5 = nn.Conv2d(in_c, in_c, k, padding=1)


    def forward(self, inputs): 
        out0 = inputs
        out1 = F.leaky_relu(self.conv1(inputs), config.lrelu_slope)
        out1 = out0 + out1
        out2 = F.leaky_relu(self.conv2(out1), config.lrelu_slope)
        out2 = out0 + out2 + out1 
        out3 = F.leaky_relu(self.conv2(out2), config.lrelu_slope)
        out3 = out0 + out2 + out3
        out4 = F.leaky_relu(self.conv2(out3), config.lrelu_slope)
        out4 = out2 + out0 + out4 + out1 

        out5 = self.conv5(out4)
        
        return out5



class Generator(nn.Module): 

    def __init__(self, input_size): 
        super(Generator, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 3, padding=1)

        self.rrdb1 = RRDB(64, 3)
        self.rrdb2 = RRDB(64, 3)
        self.rrdb3 = RRDB(64, 3)
        self.rrdb4 = RRDB(64, 3)
        self.rrdb5 = RRDB(64, 3)
        self.rrdb6 = RRDB(64, 3)
        self.rrdb7 = RRDB(64, 3)
        self.rrdb8 = RRDB(64, 3)
        self.rrdb9 = RRDB(64, 3)
        self.rrdb10 = RRDB(64, 3)
        self.prelu = nn.PReLU()
        self.conv2 = torch.nn.Conv2d(64, 64, 3, padding=1)
        
        upsamplings = []

        upsamplings.append(nn.Conv2d(64, 256, 3, padding=1))
        upsamplings.append(nn.PixelShuffle(2))
        upsamplings.append(nn.PReLU())
        upsamplings.append(nn.Conv2d(64, 256, 3, padding=1))
        upsamplings.append(nn.PixelShuffle(2))
        upsamplings.append(nn.PReLU())

        self.upsampling_block = nn.Sequential(*upsamplings)
        
        self.conv3 = nn.Conv2d(64, 3, 3, padding=1)


    def forward(self, inputs): 
        out0 = self.prelu(self.conv1(inputs))
        block1 = self.rrdb1(out0) * config.residual_scaling
        block1 = block1 + out0
        block2 = self.rrdb2(block1) * config.residual_scaling
        block2 = block1 + block2
        block3 = self.rrdb3(block2) * config.residual_scaling 
        block3 = block2 + block3 
        block4 = self.rrdb4(block3) * config.residual_scaling
        block4 = block3 + block4 
        block5 = self.rrdb5(block4) * config.residual_scaling
        block5 = block4 + block5 
        block6 = self.rrdb6(block5) * config.residual_scaling
        block6 = block5 + block6 
        block7 = self.rrdb7(block6) * config.residual_scaling
        block7 = block6 + block7 
        block8 = self.rrdb8(block7) * config.residual_scaling
        block8 = block4 + block8 
        block9 = self.rrdb9(block8) * config.residual_scaling
        block9 = block4 + block9 
        block10 = self.rrdb10(block9) * config.residual_scaling
        block10 = block4 + block10 

        out1 = self.prelu(self.conv2(block10))
        out1 = out1 + out0 
        out2 = self.upsampling_block(out1)
        
        out3 = self.conv3(out2)

        return out3
        


class DiscConvBlock(nn.Module): 
    
    def __init__(self, in_c, out_c, stride=1): 
        super(DiscConvBlock, self).__init__()
        self.in_c = in_c
        self.stride=stride
        self.out_c = out_c
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride)
        self.bn = nn.BatchNorm2d(out_c)
        self.lrelu = nn.LeakyReLU(0.2)
        
    def forward(self, x): 
        x = self.conv1(x) 
        x = self.bn(x) 
        x = self.lrelu(x)
        
        return x

class Discriminator(nn.Module): 
    
    def __init__(self): 
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3) 
        self.lrelu = nn.LeakyReLU(0.2)
                
        self.convblocks = nn.Sequential(*[
            DiscConvBlock(64, 128, 2), 
            DiscConvBlock(128, 128, 1), 
            DiscConvBlock(128, 256, 2), 
            DiscConvBlock(256, 256, 1), 
            DiscConvBlock(256, 512, 2),
            DiscConvBlock(512, 128, 1), 
        ])
        self.fc = nn.Linear(128 * 11 * 11, 1024) 
        self.lrelu = nn.LeakyReLU(0.2) 
        self.final = nn.Linear(1024, 1) 
        
    def forward(self, x): 
        x = self.conv1(x) 
        x = self.lrelu(x)
        x = self.convblocks(x) 
        x = torch.flatten(x, 1)
        x = self.fc(x) 
        x = self.lrelu(x) 
        x = self.final(x) 
        
        return x


class RelativisticDiscriminator(nn.Module):

    def __init__(self, discriminator): 
        super(RelativisticDiscriminator, self).__init__()
        self.discriminator = discriminator

    def forward(self, focus_data, compare_data):
        """
            compute the relativistic discriminator output. 
            Arguments: 
                focus_data : quantity on the left side of the difference
                compare_data : quantity on the right side of the difference
        """

        focus_output = self.discriminator(focus_data)
        compare_output = self.discriminator(compare_data)

        difference = focus_output - compare_data.mean(dim=0)

        return difference
        

class ContentLoss(nn.Module): 


    def __init__(self): 
        super(ContentLoss, self).__init__()
        self.vgg19 = torchvision.models.vgg19(pretrained=True).eval()
        self.layers = nn.Sequential(*list(self.vgg19.features.children())[:36])
        self.loss = torch.nn.MSELoss()

    def forward(self, sr, hr): 
        sr_vgg = self.layers(sr)
        hr_vgg = self.layers(hr)

        return self.loss(sr_vgg, hr_vgg)