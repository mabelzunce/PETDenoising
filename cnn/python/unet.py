import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class AddResidual(nn.Module):
    def __init__(self):
        super(AddResidual, self).__init__()
        #self.relu = nn.ReLU(inplace=True)
    def forward(self, x, residual):
        x = x + residual
        # Image has only positive values:

        # x = self.relu(x) # Removing the relu as it is creating too many 0 zero values as negative values are all penalized in the same way.
        return x

class Unet(nn.Module):
    def __init__(self, in_channels, classes):
        super(Unet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = classes

        self.inc = InConv(in_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        self.down4 = Down(128, 256)
        self.up1 = Up(256+128, 128)
        self.up2 = Up(128+64, 64)
        self.up3 = Up(64+32, 32)
        self.up4 = Up(32+16, 16)
        self.outc = OutConv(16, classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

class Unet512(nn.Module):
    def __init__(self, in_channels, classes,out_first_layer):
        super(Unet512, self).__init__()
        self.n_channels = in_channels
        self.n_classes = classes
        self.n_out_first_layer = out_first_layer

        self.inc = InConv(in_channels, out_first_layer)

        self.down1 = Down(out_first_layer, out_first_layer * 2)
        self.down2 = Down(out_first_layer * 2, out_first_layer * 4)
        self.down3 = Down(out_first_layer * 4, out_first_layer * 8)
        self.down4 = Down(out_first_layer * 8,  out_first_layer * 16)

        self.up1 = Up((out_first_layer * 16) + (out_first_layer * 8), out_first_layer * 8)
        self.up2 = Up((out_first_layer * 8)+(out_first_layer * 4), out_first_layer * 4)
        self.up3 = Up((out_first_layer * 4) + (out_first_layer * 2), out_first_layer * 2)
        self.up4 = Up((out_first_layer * 2) + out_first_layer, out_first_layer )

        self.outc = OutConv(out_first_layer, classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

class UnetDe1a16Hasta512(nn.Module):
    def __init__(self, in_channels, classes,out_first_layer):
        super(UnetDe1a16Hasta512, self).__init__()
        self.n_channels = in_channels
        self.n_classes = classes
        self.n_out_first_layer = out_first_layer

        self.inc = InConv(in_channels, out_first_layer)

        self.down1 = Down(out_first_layer, out_first_layer * 2)
        self.down2 = Down(out_first_layer * 2, out_first_layer * 4)
        self.down3 = Down(out_first_layer * 4, out_first_layer * 8)
        self.down4 = Down(out_first_layer * 8,  out_first_layer * 16)
        self.down5 = Down(out_first_layer * 16, out_first_layer * 32)

        self.up1 = Up((out_first_layer * 32) + (out_first_layer * 16), out_first_layer * 16)
        self.up2 = Up((out_first_layer * 16) + (out_first_layer * 8), out_first_layer * 8)
        self.up3 = Up((out_first_layer * 8) + (out_first_layer * 4), out_first_layer * 4)
        self.up4 = Up((out_first_layer * 4) + (out_first_layer * 2), out_first_layer * 2)
        self.up5 = Up((out_first_layer * 2) + out_first_layer, out_first_layer)

        self.outc = OutConv(out_first_layer, classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)

        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        x = self.outc(x)
        return x

class UnetWithResidual(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetWithResidual, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels

        self.inc = InConv(in_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        self.down4 = Down(128, 256)
        self.up1 = Up(256 + 128, 128)
        self.up2 = Up(128 + 64, 64)
        self.up3 = Up(64 + 32, 32)
        self.up4 = Up(32 + 16, 16)
        self.outc = OutConv(16, 1)
        self.add_res = AddResidual()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        y4 = self.up1(x5, x4)
        y3 = self.up2(y4, x3)
        y2 = self.up3(y3, x2)
        y1 = self.up4(y2, x1)
        res = self.outc(y1)
        y = self.add_res(x, res)
        return y

class UnetWithResidual5Layers(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetWithResidual5Layers, self).__init__()
        self.n_channels = in_channels
        self.n_classes =  out_channels

        self.inc = InConv(in_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        self.down4 = Down(128, 256)
        self.down5 = Down(256, 512)
        self.up1 = Up(512 + 256, 256)
        self.up2 = Up(256 + 128, 128)
        self.up3 = Up(128 + 64, 64)
        self.up4 = Up(64 + 32, 32)
        self.up5 = Up(32 + 16, 16)
        self.outc = OutConv(16, 1)
        self.add_res = AddResidual()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        y5 = self.up1(x6, x5)
        y4 = self.up2(y5, x4)
        y3 = self.up3(y4, x3)
        y2 = self.up4(y3, x2)
        y1 = self.up5(y2, x1)
        res = self.outc(y1)
        y = self.add_res(x, res)
        return y