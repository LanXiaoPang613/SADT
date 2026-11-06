"""CIFAR architectures used in the SADT experiments."""

import torch.nn as nn
import torch.nn.functional as F


class ConvBranch(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, inputs):
        return self.layers(inputs)


class CIFAR10CNN(nn.Module):
    """Eight-layer CNN used by the released CIFAR-10 protocol."""

    def __init__(self) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            ConvBranch(3, 64), ConvBranch(64, 64), nn.MaxPool2d(2)
        )
        self.block2 = nn.Sequential(
            ConvBranch(64, 128), ConvBranch(128, 128), nn.MaxPool2d(2)
        )
        self.block3 = nn.Sequential(
            ConvBranch(128, 196), ConvBranch(196, 196), nn.MaxPool2d(2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(4 * 4 * 196, 256), nn.BatchNorm1d(256), nn.ReLU()
        )
        self.fc2 = nn.Linear(256, 10)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(
                    module.weight, mode="fan_in", nonlinearity="relu"
                )
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, inputs):
        features = self.block3(self.block2(self.block1(inputs)))
        features = self.fc1(features.reshape(features.shape[0], -1))
        return self.fc2(features)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, inputs):
        output = F.relu(self.bn1(self.conv1(inputs)))
        output = self.bn2(self.conv2(output)) + self.shortcut(inputs)
        return F.relu(output)


class CIFARResNet(nn.Module):
    def __init__(self, blocks: tuple[int, int, int, int], classes: int) -> None:
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, blocks[0], 1)
        self.layer2 = self._make_layer(128, blocks[1], 2)
        self.layer3 = self._make_layer(256, blocks[2], 2)
        self.layer4 = self._make_layer(512, blocks[3], 2)
        self.linear = nn.Linear(512, classes)
        self._reset_parameters()

    def _make_layer(self, planes: int, count: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (count - 1)
        layers = []
        for layer_stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, layer_stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(
                    module.weight, mode="fan_in", nonlinearity="relu"
                )
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, inputs):
        output = F.relu(self.bn1(self.conv1(inputs)))
        output = self.layer4(self.layer3(self.layer2(self.layer1(output))))
        output = F.avg_pool2d(output, 4)
        return self.linear(output.reshape(output.shape[0], -1))


def resnet34(classes: int = 100) -> CIFARResNet:
    return CIFARResNet((3, 4, 6, 3), classes)
