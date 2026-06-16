import torch.nn as nn

class FullyConvNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        # Encoder (Convolutional Layers)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1),  # 256 -> 128
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1), # 128 -> 64
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1), # 64 -> 32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 32 -> 16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 16 -> 8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # 8 -> 4
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), # 4 -> 2
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1), # 2 -> 1
            nn.ReLU(inplace=True)   # 瓶颈层，可省略 BatchNorm
        )

        # Decoder (Deconvolutional Layers)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1), # 1 -> 2
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # 2 -> 4
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 4 -> 8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 8 -> 16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # 16 -> 32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.deconv6 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), # 32 -> 64
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.deconv7 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1), # 64 -> 128
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        self.deconv8 = nn.Sequential(
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1), # 128 -> 256
            nn.Tanh()   # 输出归一化到 [-1, 1]
        )

    def forward(self, x):
        # Encoder forward pass
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)   # 瓶颈 1×1

        # Decoder forward pass
        y1 = self.deconv1(x8)
        y2 = self.deconv2(y1)
        y3 = self.deconv3(y2)
        y4 = self.deconv4(y3)
        y5 = self.deconv5(y4)
        y6 = self.deconv6(y5)
        y7 = self.deconv7(y6)
        output = self.deconv8(y7)   # 256×256×3

        return output    