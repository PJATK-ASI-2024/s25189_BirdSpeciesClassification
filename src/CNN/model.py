import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class BirdCNNModel(nn.Module):
    def __init__(self, num_classes=200):
        super(BirdCNNModel, self).__init__()
        # Simple CNN example:
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        # Input (224,244), after 3 poolings -> (224/2 = 112, /2 = 56, /2 = 28) -> 28x28 Feature map
        # 128 channels * 28 * 28 = 128 * 784 = 100,352 features

        self.classifier = nn.Sequential(
            nn.Linear(128*28*28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x 

class BirdResNetModel(nn.Module):
    def __init__(self, num_classes=200):
        super(BirdResNetModel, self).__init__()
        self.model = models.resnet50(pretrained=True)  # Load pre-trained ResNet50
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)  # Replace final layer

    def forward(self, x):
        return self.model(x)       