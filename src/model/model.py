import torch
import torch.nn as nn
from torchvision import models

class DogCatClassifier(nn.Module):
    def __init__(self, num_classes=1):
        super(DogCatClassifier, self).__init__()
        # Load the pre-trained VGG-16 model
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        # Freeze all parameters in the VGG16 feature extractor
        for param in vgg16.features.parameters():
            param.requires_grad = False

        # Replace the classifier (top) layers
        self.features = vgg16.features
        self.avgpool = vgg16.avgpool
        
        # Custom classifier for dog/cat classification
        self.classifier = nn.Sequential(
            nn.Linear(25088, 512),  # VGG-16 outputs 25088 features before the original classifier
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
            nn.Sigmoid()  # Sigmoid for binary classification
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x 