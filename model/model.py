
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

import torch.nn as nn
import torch
from torchvision import models

class CombinedDenseNetEfficientNet(nn.Module):
    def __init__(self, num_classes=2):
        super(CombinedDenseNetEfficientNet, self).__init__()
        
        # Load a pretrained DenseNet model
        self.densenet = models.densenet121(pretrained=True)
        num_ftrs_densenet = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_ftrs_densenet, num_classes)
        
        # Load a pretrained EfficientNet model
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        num_ftrs_efficientnet = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Linear(num_ftrs_efficientnet, num_classes)
    
    def forward(self, x):
        # Pass input through both DenseNet and EfficientNet models
        densenet_output = self.densenet(x)
        efficientnet_output = self.efficientnet(x)
        
        # Combine the outputs (for simplicity, let's average them here)
        combined_output = (densenet_output + efficientnet_output) / 2
        
        return combined_output
