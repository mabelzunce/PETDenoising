import torch
from torch import nn

# Unet CLASICA (nuevo formato)

class DownConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DownConv, self).__init__()
        self.DownLayer = nn.Sequential(

            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.DownLayer(x)
        return x


class MaxPool(nn.Module):
    def __init__(self):
        super(MaxPool, self).__init__()
        self.Pool = torch.nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.Pool(x)
        return x


class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()

        self.ConvTransp = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.ConvTransp = torch.nn.ConvTranspose2d((in_channels//2), out_channels, kernel_size = 2, stride = 2, padding = 0)
        self.UpConv = DownConv(in_channels, out_channels)

    def forward(self, xAnt, xDown):
        layerConvTransposed = self.ConvTransp(xAnt)
        concat = torch.cat([layerConvTransposed, xDown], dim=1)
        x = self.UpConv(concat)

        return x


class OutUnet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutUnet, self).__init__()
        self.OutUnet = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.OutUnet(x)
        return x


class Unet(nn.Module):

    def __init__(self):
        super(Unet, self).__init__()

        # Contract
        self.Layer1Down = DownConv(1, 32)
        self.Layer2Down = DownConv(32, 64)
        self.Layer3Down = DownConv(64, 128)
        self.Layer4Down = DownConv(128, 256)
        self.Layer5Down = DownConv(256, 512)

        self.Middle = DownConv(512, 512)

        self.Layer1Up = UpConv(1024, 256)
        self.Layer2Up = UpConv(512, 128)
        self.Layer3Up = UpConv(256, 64)
        self.Layer4Up = UpConv(128, 64)
        self.Layer5Up = UpConv(64 + 32, 32)

        self.MaxPool = MaxPool()

        self.Out = OutUnet(32, 1)

    def forward(self, x):
        # Down
        conv1 = self.Layer1Down(x)
        maxPool1 = self.MaxPool(conv1)

        conv2 = self.Layer2Down(maxPool1)
        maxPool2 = self.MaxPool(conv2)

        conv3 = self.Layer3Down(maxPool2)
        maxPool3 = self.MaxPool(conv3)

        conv4 = self.Layer4Down(maxPool3)
        maxPool4 = self.MaxPool(conv4)

        conv5 = self.Layer5Down(maxPool4)
        maxPool5 = self.MaxPool(conv5)

        middle = self.Middle(maxPool5)

        # Up
        up1 = self.Layer1Up(middle, conv5)
        up2 = self.Layer2Up(up1, conv4)
        up3 = self.Layer3Up(up2, conv3)
        up4 = self.Layer4Up(up3, conv2)
        up5 = self.Layer5Up(up4, conv1)

        outUNet = self.Out(up5)

        return outUNet