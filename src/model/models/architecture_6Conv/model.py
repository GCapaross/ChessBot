import torch.nn as nn
import torch.nn.functional as F

class CompleteChessBotNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 13 channels: 12 pieces + side to play (tensor[true's|false's])

        self.inputLayer = nn.Conv2d(13, 64, kernel_size=3, padding='same')
        self.batchnorm0 = nn.BatchNorm2d(64)

        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, padding='same')
        self.batchnorm1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding='same')
        self.batchnorm2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding='same')
        self.batchnorm3 = nn.BatchNorm2d(64)


        self.convLayers = nn.Sequential(
            self.inputLayer,
            self.batchnorm0,
            nn.ReLU(),
            self.conv1,
            self.batchnorm1,
            nn.ReLU(),
            self.conv2,
            self.batchnorm2,
            nn.ReLU(),
            self.conv3,
            self.batchnorm3,
            nn.ReLU()
        )

        self.fc1 = nn.Linear(8 * 8 * 64, 256) # 8 * 8 (num of squares) * 128 (num of channels)
        self.fc2 = nn.Linear(256, 64 * 63) # (Choose 2 squares from the board where the order matters)
        
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        # Initialize weights
        nn.init.kaiming_uniform_(self.inputLayer.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv3.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)


    def forward(self, x):
        x = self.convLayers(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # Output raw logits
        return x