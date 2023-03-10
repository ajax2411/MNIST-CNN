
import torch.nn as nn
from torch.nn.functional import relu, log_softmax


class CNN(nn.Module):

    def __init__(self, in_channels, num_classes, conv_kernel_size, pool_kernel_size, conv_stride, pool_stride, fc1_output):

        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 10, kernel_size=conv_kernel_size, stride=conv_stride)
        self.conv2 = nn.Conv2d(10, 24, kernel_size=conv_kernel_size, stride=conv_stride)

        self.pool = nn.MaxPool2d(pool_kernel_size, stride=pool_stride)

        self.fc1 = nn.Linear(24 * 4 * 4, fc1_output)
        self.fc2 = nn.Linear(fc1_output, fc1_output * 2)
        self.fc3 = nn.Linear(fc1_output * 2, fc1_output * 4)
        self.fc4 = nn.Linear(fc1_output * 4, num_classes)

    def forward(self, x):

        x = self.conv1(x)
        x = relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = relu(x)
        x = self.pool(x)

        x = x.view(-1, 24 * 4 * 4)

        x = self.fc1(x)
        x = relu(x)

        x = self.fc2(x)
        x = relu(x)

        x = self.fc3(x)
        x = relu(x)

        x = self.fc4(x)

        return x