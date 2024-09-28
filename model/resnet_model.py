
import torch.nn as nn
import torch
from torchvision import models

class ResNetModel(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNetModel, self).__init__()
        
        # Load a pretrained ResNet model
        self.resnet = models.resnet50(pretrained=True)
        num_ftrs_resnet = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs_resnet, num_classes)
    
    def forward(self, x):
        return self.resnet(x)
