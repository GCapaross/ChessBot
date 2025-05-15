from torch import nn
import torch.nn.functional as F

class _CompleteChessBotNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 13 channels: 12 pieces + side to play (tensor[true's|false's])
        self.conv1 = nn.Conv2d(13, 64, kernel_size=3, padding='same')
        self.batchnorm1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding='same')
        self.batchnorm2 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(8 * 8 * 128, 256) # 8 * 8 (num of squares) * 128 (num of channels)
        self.fc2 = nn.Linear(256, 64 * 63) # (Choose 2 squares from the board where the order matters)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten()

        # Initialize weights
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.batchnorm1(x)

        x = self.relu(self.conv2(x))
        x = self.batchnorm2(x)

        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # Output raw logits
        return x