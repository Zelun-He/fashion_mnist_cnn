import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ------------------ ConvBlock ------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

# ------------------ BaseNet ------------------
class BaseNet(nn.Module):
    def __init__(self, version=10):
        super(BaseNet, self).__init__()
        def make_module(in_c, out_c, num_convs, first_stride=1):
            layers = [ConvBlock(in_c, out_c, stride=first_stride)]
            for _ in range(num_convs - 1):
                layers.append(ConvBlock(out_c, out_c))
            return nn.Sequential(*layers)

        convs_per_module = 2 if version == 10 else 4

        self.module1 = make_module(1, 8, convs_per_module)
        self.module2 = make_module(8, 16, convs_per_module, first_stride=2)
        self.module3 = make_module(16, 32, convs_per_module, first_stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = self.module1(x)
        x = self.module2(x)
        x = self.module3(x)
        x = self.global_pool(x).view(x.size(0), -1)
        return self.fc(x)

# ------------------ ResidualBlock ------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

# ------------------ ResNet ------------------
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )

        self.layer1 = self._make_layer(8, 8, 2, stride=1)
        self.layer2 = self._make_layer(8, 16, 2, stride=2)
        self.layer3 = self._make_layer(16, 32, 2, stride=2)
        self.layer4 = self._make_layer(32, 64, 2, stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x).view(x.size(0), -1)
        return self.fc(x)

# ------------------ Data Test ------------------
transform = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor()
])

dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

images, labels = next(iter(loader))
print(f"Input batch shape: {images.shape}")

# ------------------ Model Test ------------------
basenet = BaseNet(version=10)
resnet = ResNet()

out1 = basenet(images)
out2 = resnet(images)

print(f"BaseNet output shape: {out1.shape}")  # Expected: [8, 10]
print(f"ResNet output shape: {out2.shape}")   # Expected: [8, 10]
