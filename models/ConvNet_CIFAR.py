import torch.nn as nn


class ConvNet_CIFAR(nn.Module):
    def __init__(self):
        super(ConvNet_CIFAR, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(2304, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        output = self.features(x)  # on calcule la sortie des conv
        output = output.view(128, -1)  # on applati les feature map 2D en un
        # vecteur 1D pour chaque input
        return self.classifier(output)

