
import torch.nn as nn
from torch import flatten
from torch.nn.functional import relu


class CNN(nn.Module):

    def __init__(self, in_channels, classes, kernel_size, stride, padding=0):

        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 10, kernel_size=kernel_size, stride=stride, padding=padding)

        self.pool = nn.MaxPool2d(kernel_size, stride=stride)

        self.conv2 = nn.Conv2d(10, 24, kernel_size=kernel_size, stride=stride)

        self.fc1 = nn.Linear(24 * kernel_size * kernel_size, 28)
        self.fc2 = nn.Linear(28, classes)

    def forward(self, x):

        x = self.pool(relu(self.conv1(x)))
        x = self.pool(relu(self.conv2(x)))

        x = flatten(x, 1)

        x = relu(self.fc1(x))
        x = self.fc2(x)

        return x