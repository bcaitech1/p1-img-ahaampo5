import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
from efficientnet_pytorch import EfficientNet


class PreTrainedResNet152(nn.Module):
    def __init__(self):
        super(PreTrainedResNet152,self).__init__()
        self.backbone = models.resnet152(pretrained=True)
    
    def forward(self, x):
        x = self.backbone(x)
        return x

class PreTrainedEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(PreTrainedEfficientNet,self).__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        self.num_classes = num_classes
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.5)
        self.fc = nn.Sequential(nn.Linear(1000,256),
                               nn.ReLU(),
                               nn.Dropout(0.5),
                               nn.Linear(256,num_classes))        
        self.set_freeze()



    def forward(self, x):
        x = self.backbone(x)
        x1 = self.drop1(x)
        x2 = self.drop2(x)
        x1 = self.fc(x1)
        x2 = self.fc(x2)
        x = (x1+x2)/2
        # return torch.sigmoid(x)
        return x

    def set_freeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def set_free(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
            

def make_block(r,n):
    residual = []
    
    for i in range(r):
        block = ResBlock(num_channels=n)
        residual.append(block)
    
    return nn.Sequential(*residual)


class ResBlock(nn.Module):
    def __init__(self,num_channels=16):
        super(ResBlock,self).__init__()
        
        self.conv1 = nn.Conv2d(num_channels,num_channels,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2,inplace=True)
        
        self.conv2 = nn.Conv2d(num_channels,num_channels,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
    
    def forward(self,x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        
        out = self.leakyrelu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        
        return out


class ResizingNetwork(nn.Module):
    def __init__(self,r=1, n=16):
        super(ResizingNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=n,kernel_size=7,stride=1,padding=3)
        self.leakyrelu1 = nn.LeakyReLU(negative_slope=0.2,inplace=True)
        
        self.conv2 = nn.Conv2d(n,n,kernel_size=1,stride=1)
        self.leakyrelu2 = nn.LeakyReLU(negative_slope=0.2,inplace=True)
        self.bn1 = nn.BatchNorm2d(n)
        
        
        self.resblock = make_block(r,n)
        
        
        self.conv3 = nn.Conv2d(n,n,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(n)
        
        self.conv4 = nn.Conv2d(n,out_channels=3,kernel_size=7,stride=1,padding=3)
        
    
    def forward(self,x):
        
        residual = F.interpolate(x,scale_factor=0.5,mode='bilinear')
        
        out = self.conv1(x)
        out = self.leakyrelu1(out)
        
        out = self.conv2(out)
        out = self.leakyrelu2(out)
        out = self.bn1(out)
        
        out_residual = F.interpolate(out,scale_factor=0.5,mode='bilinear')
        
        out = self.resblock(out_residual)
        
        out = self.conv3(out)
        out = self.bn2(out)
        out += out_residual
        
        out = self.conv4(out)
        out += residual
        
        return out

