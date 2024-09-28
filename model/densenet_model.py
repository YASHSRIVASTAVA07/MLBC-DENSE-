
import torch.nn as nn
import torch
from torchvision import models

class DenseNetModel(nn.Module):
    def __init__(self, num_classes=2):
        super(DenseNetModel, self).__init__()
        
        # Load a pretrained DenseNet model
        self.densenet = models.densenet121(pretrained=True)
        num_ftrs_densenet = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_ftrs_densenet, num_classes)
    
    def forward(self, x):
        return self.densenet(x)
