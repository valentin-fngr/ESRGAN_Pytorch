from turtle import forward
import torch 
import numpy as np 
import torch.nn as nn 
import torch.nn.functional as F
import config
import torchvision


class DenselyConnectedLayer(nn.Module): 
    """
        RRDB block ( https://arxiv.org/pdf/1608.06993.pdf )
    """ 
    
    def __init__(self, in_c, k_growth, kernel_size=3): 
        super(DenselyConnectedLayer, self).__init__() 
        self.conv1 = nn.Conv2d(in_c , k_growth, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(in_c + k_growth, k_growth, kernel_size, padding=1)
        self.conv3 = nn.Conv2d(in_c + 2*k_growth,k_growth, kernel_size, padding=1)
        self.conv4 = nn.Conv2d(in_c +  3*k_growth, k_growth, kernel_size, padding=1)
        self.conv5 = nn.Conv2d(in_c +  4*k_growth, in_c, kernel_size, padding=1)


    def forward(self, inputs): 
        out0 = inputs
        out1 = F.leaky_relu(self.conv1(out0), config.lrelu_slope)
        out2 = F.leaky_relu(self.conv2(torch.cat([out0, out1], dim=1)), config.lrelu_slope)
        out3 = F.leaky_relu(self.conv3(torch.cat([out0, out1, out2], dim=1)), config.lrelu_slope)
        out4 = F.leaky_relu(self.conv4(torch.cat([out0, out1, out2, out3], dim=1)), config.lrelu_slope)
        out5 = self.conv5(torch.cat([out0, out1, out2, out3, out4], dim=1))

        return out5


class RRDB(nn.Module): 

    def __init__(self, in_c, k_growth): 
        super(RRDB, self).__init__()
        self.layer1 = DenselyConnectedLayer(in_c, k_growth)
        self.layer2 = DenselyConnectedLayer(in_c, k_growth)
        self.layer3 = DenselyConnectedLayer(in_c, k_growth)
        self.layer4 = DenselyConnectedLayer(in_c, k_growth)

    def forward(self, inputs): 
        out1 = self.layer1(inputs) * config.residual_scaling
        out2 = self.layer2(inputs) * config.residual_scaling
        out3 = self.layer3(inputs) * config.residual_scaling
        out4 = self.layer4(inputs) * config.residual_scaling
        output = out1 + out2 + out3 + out4

        return output


class Generator(nn.Module): 

    def __init__(self, input_size): 
        super(Generator, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 3, padding=1)

        self.rrdbs = nn.Sequential(*[RRDB(64, 32) for i in range(16)])
        
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
        rrdb_pass = self.rrdbs(out0)
        out1 = self.prelu(self.conv2(rrdb_pass))
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
        vgg19 = torchvision.models.vgg19(pretrained=True).eval()
        self.layers = nn.Sequential(*list(vgg19.features.children())[:36])
        for param in self.layers.parameters():
            param.requires_grad = False
        self.loss = torch.nn.MSELoss()

    def forward(self, sr, hr): 
        sr_vgg = self.layers(sr)
        hr_vgg = self.layers(hr)

        return self.loss(sr_vgg, hr_vgg)