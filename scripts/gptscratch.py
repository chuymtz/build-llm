import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock3D, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class Bottleneck3D(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck3D, self).__init__()
        self.conv1 = conv1x1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet3D(nn.Module):
    """
    3D ResNet model for volumetric data.
    This class implements a 3D version of the ResNet architecture, suitable for processing volumetric data such as medical images or video sequences. 
    The network consists of an initial 3D convolutional layer, followed by four residual layers, global average pooling, and a fully connected output layer.
    Args:
        block (nn.Module): Residual block class to be used (e.g., BasicBlock3D or Bottleneck3D).
        layers (list of int): Number of blocks in each of the four residual layers.
        in_channels (int, optional): Number of input channels. Default is 1.
        out_channels (int, optional): Number of output channels (e.g., number of classes for classification). Default is 1.
    Attributes:
        inplanes (int): Number of channels for the first convolutional layer.
        conv1 (nn.Conv3d): Initial 3D convolutional layer.
        bn1 (nn.BatchNorm3d): Batch normalization after the initial convolution.
        relu (nn.ReLU): ReLU activation function.
        maxpool (nn.MaxPool3d): Max pooling layer after the initial convolution.
        layer1-4 (nn.Sequential): Four sequential layers of residual blocks.
        avgpool (nn.AdaptiveAvgPool3d): Adaptive average pooling to reduce spatial dimensions to 1x1x1.
        fc (nn.Linear): Fully connected layer for output.
    Methods:
        _make_layer(block, planes, blocks, stride=1):
            Constructs a sequential layer of residual blocks, with optional downsampling for dimension matching.
        forward(x):
            Defines the forward pass of the network. Applies convolution, normalization, activation, pooling, residual layers, global pooling, and the final fully connected layer to input tensor x.
    Example:
        >>> model = ResNet3D(BasicBlock3D, [2, 2, 2, 2], in_channels=1, out_channels=10)
        >>> output = model(torch.randn(8, 1, 32, 32, 32))  # batch of 8, 1 channel, 32x32x32 volumes
    """
    def __init__(self, block, layers, in_channels=1, out_channels=1):
        super(ResNet3D, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, out_channels)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def resnet3d_32(in_channels=1, out_channels=1):
    return ResNet3D(BasicBlock3D, [3, 4, 6, 3], in_channels, out_channels)

def resnet3d_50(in_channels=1, out_channels=1):
    return ResNet3D(Bottleneck3D, [3, 4, 6, 3], in_channels, out_channels)

# Example usage:
model = resnet3d_32(in_channels=1, out_channels=1)
x = torch.randn(2, 1, 32, 32, 32)  # batch of 2, 1 channel, 32x32x32 patch
y = model(x)  # output shape: (2, 1)


class RandomVolumetricDataset(Dataset):
    """
    Example dataset for 3D volumetric data.
    Generates random 3D volumes and random targets for demonstration.
    """
    def __init__(self, num_samples=100, in_channels=1, volume_shape=(32, 32, 32), out_channels=1):
        self.num_samples = num_samples
        self.in_channels = in_channels
        self.volume_shape = volume_shape
        self.out_channels = out_channels

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = np.random.randn(self.in_channels, *self.volume_shape).astype(np.float32)
        y = np.random.randn(self.out_channels).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)

# Parameters
in_channels = 1
out_channels = 1
batch_size = 4

# Create dataset and dataloader
dataset = RandomVolumetricDataset(num_samples=50, in_channels=in_channels, out_channels=out_channels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Example iteration
for batch_x, batch_y in dataloader:
    print("Batch X shape:", batch_x.shape)  # (batch_size, in_channels, D, H, W)
    print("Batch Y shape:", batch_y.shape)  # (batch_size, out_channels)


    # Training routine
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet3d_32(in_channels=in_channels, out_channels=out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    num_epochs = 5

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")



