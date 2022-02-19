import torch 
import numpy as np 
import torch.nn as nn 
import torch.nn.functional as F
import config

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

        out1 = self.prelu(self.conv2(block5))
        out1 = out1 + out0 
        out2 = self.upsampling_block(out1)
        
        out3 = self.conv3(out2)

        return out3
        