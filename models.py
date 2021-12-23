import torch
import torch.nn as nn
from collections import OrderedDict

from layers import *


class MAMLConvNet(nn.Module):
    def __init__(self, n_way):
        super(MAMLConvNet, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=32, track_running_stats=False)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=32, track_running_stats=False)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=32, track_running_stats=False)
        self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=32, track_running_stats=False)
        self.mp4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 5 * 5, n_way)
        
    def forward(self, x, weights=None):
        if weights is None:
            x = self.mp1(self.act(self.bn1(self.conv1(x))))
            x = self.mp2(self.act(self.bn2(self.conv2(x))))
            x = self.mp3(self.act(self.bn3(self.conv3(x))))
            x = self.mp4(self.act(self.bn4(self.conv4(x))))
            x = self.fc(self.flatten(x))
        else:
            x = conv2d(x, weights['conv1.weight'], weights['conv1.bias'], 1, 1)
            x = batchnorm(x, weights['bn1.weight'], weights['bn1.bias'], 1)
            x = torch.nn.functional.relu(x)
            x = maxpool(x, 2, 2) 
            
            x = conv2d(x, weights['conv2.weight'], weights['conv2.bias'], 1, 1)
            x = batchnorm(x, weights['bn2.weight'], weights['bn2.bias'], 1)
            x = torch.nn.functional.relu(x)
            x = maxpool(x, 2, 2) 
            
            x = conv2d(x, weights['conv3.weight'], weights['conv3.bias'], 1, 1)
            x = batchnorm(x, weights['bn3.weight'], weights['bn3.bias'], 1)
            x = torch.nn.functional.relu(x)
            x = maxpool(x, 2, 2) 
            
            x = conv2d(x, weights['conv4.weight'], weights['conv4.bias'], 1, 1)
            x = batchnorm(x, weights['bn4.weight'], weights['bn4.bias'], 1)
            x = torch.nn.functional.relu(x)
            x = maxpool(x, 2, 2) 
            
            x = x.view(x.size(0), -1)
            x = linear(x, weights['fc.weight'], weights['fc.bias'])
        return x