import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torchvision import transforms,models

class Encoder(nn.Module):
    def __init__(self):
        super (Encoder, self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        '''
        for param in resnet18.parameters():
            param.requires_grad_(False)
        '''
        self.layer1 = nn.Sequential(*list(resnet18.children())[:5])
        self.layer2 = nn.Sequential (*list(resnet18.children())[5:6])
        self.layer3 = nn.Sequential(*list(resnet18.children())[6:7])
        self.layer4 = nn.Sequential(*list(resnet18.children())[7:8])
    def forward(self,x):
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return layer1,layer2,layer3,layer4

class BasicBlock(nn.Module):
    def __init__(self,ip_chnl, op_chnl):
        super(BasicBlock,self).__init__()
        self.conv1 = nn.Conv2d(ip_chnl,op_chnl,3,padding=1)
        self.bn1 = nn.BatchNorm2d(op_chnl)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(op_chnl,op_chnl,3,padding=1)
        self.bn2 = nn.BatchNorm2d(op_chnl)
    def forward (self,x):
        #ip = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        #x += ip
        return x 
    
class Upsample(nn.Module):
    def __init__(self,ip_chnl,op_chnl,filter_size=(3,3),stride=2,padding=(0,0),output_padding= (0,0)):
        super (Upsample,self).__init__()
        self.convT = nn.ConvTranspose2d(ip_chnl,op_chnl,filter_size,stride=stride,
                               padding=padding,output_padding=output_padding) 
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(op_chnl)
        self.conv = BasicBlock(op_chnl,op_chnl)
    def forward(self,x):
        x = self.convT(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        return x
    
class FPN(nn.Module):
    def __init__(self,ip_chnl,upsample):
        super (FPN,self).__init__()
        self.conv1 = nn.Conv2d(ip_chnl,64,3,padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=upsample)
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.upsample(x)
        return x
class Unet(nn.Module):
    def __init__(self,ip_channel,op_channel):
        super(Unet,self).__init__()
        self.downsample = Encoder()
        self.up1 = Upsample(512,256,padding=(1,1),output_padding=(1,1))
        self.up2 = Upsample(256,128,padding=(1,1),output_padding=(1,1))
        self.up3 = Upsample(128,64,padding=(1,1),output_padding=(1,1))
        self.up4 = Upsample (64,64,padding=(1,1),output_padding=(1,1))
        self.fpn1 = FPN(256,8)
        self.fpn2 = FPN(128,4)
        self.fpn3 = FPN(64,2)
        self.fpn4 = FPN(64,1)
        self.up5 = Upsample(256,64,filter_size=(2,2))
        self.conv1 = BasicBlock (64,32)
        self.conv2 = nn.Conv2d(32,op_channel,1)
    def forward(self, x):
        x1,x2,x3,x4 = self.downsample(x)
        #print(x1.size(), x2.size(),x3.size(),x4.size())
        x = self.up1(x4)
        #print("up1=",x.size())
        x += x3
        x3 = x
        x = self.up2(x)
        #print("up2=",x.size())
        x += x2
        x2 = x
        x = self.up3(x)
        #print("up3=",x.size())
        x +=x1
        x1 = x
        x = self.up4(x)
        #print("up4=",x.size())
        x3 = self.fpn1(x3)
        x2 = self.fpn2(x2)
        x1 = self.fpn3(x1)
        x = self.fpn4(x)
        x = torch.cat((x,x1,x2,x3),1)
        x = self.up5(x)
        #print(x.size())
        x = self.conv1(x)
        #print("up5=",x.size())
        x = self.conv2(x)
        #print(x.size())
        return x
 