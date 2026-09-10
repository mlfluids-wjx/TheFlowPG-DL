"""SRM backbone: a convolutional encoder-decoder with skip connections."""
import torch
from torch import nn
import torch.nn.functional as F


def make_norm(norm, num_channels, gn_groups=8):
    """
    norm: None / "bn" / "gn" / "ln"
    - "ln": implemented as GroupNorm(1, C); independent of H and W
    """
    if norm is None or norm == "none":
        return nn.Identity()
    norm = norm.lower()
    if norm == "bn":
        return nn.BatchNorm2d(num_channels)
    if norm == "gn":
        g = min(gn_groups, num_channels)
        while g > 1 and (num_channels % g != 0):
            g -= 1
        return nn.GroupNorm(g, num_channels)
    if norm == "ln":
        return nn.GroupNorm(1, num_channels)
    raise ValueError(f"Unknown norm='{norm}', choose from None/'bn'/'gn'/'ln'.")

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, norm="none", gn_groups=8):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            make_norm(norm, out_ch, gn_groups),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            make_norm(norm, out_ch, gn_groups),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        self.up_scale = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x1, x2):
        x2 = self.up_scale(x2)

        diffY = x1.size()[2] - x2.size()[2]
        diffX = x1.size()[3] - x2.size()[3]

        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return x

class DownLayer(nn.Module):
    def __init__(self, in_ch, out_ch, norm="none", gn_groups=8):
        super(DownLayer, self).__init__()
        self.pool = nn.MaxPool2d(2, stride=2, padding=0)
        self.conv = DoubleConv(in_ch, out_ch, norm=norm, gn_groups=gn_groups)

    def forward(self, x):
        return self.conv(self.pool(x))

class UpLayer(nn.Module):
    def __init__(self, in_ch, out_ch, norm="none", gn_groups=8):
        super(UpLayer, self).__init__()
        self.up = Up(in_ch, out_ch)
        self.conv = DoubleConv(in_ch, out_ch, norm=norm, gn_groups=gn_groups)

    def forward(self, x1, x2):
        a = self.up(x1, x2)
        return self.conv(a)


class UNet(nn.Module):
    def __init__(self, dimensions=1, norm="none", gn_groups=8):
        super(UNet, self).__init__()
        
        self.conv1 = DoubleConv(dimensions, 32, norm=norm, gn_groups=gn_groups)
        self.down1 = DownLayer(32, 64, norm=norm, gn_groups=gn_groups)
        self.down2 = DownLayer(64, 128, norm=norm, gn_groups=gn_groups)
        self.down3 = DownLayer(128, 256, norm=norm, gn_groups=gn_groups)
        self.down4 = DownLayer(256, 512, norm=norm, gn_groups=gn_groups)
        self.up1 = UpLayer(512, 256, norm=norm, gn_groups=gn_groups)
        self.up2 = UpLayer(256, 128, norm=norm, gn_groups=gn_groups)
        self.up3 = UpLayer(128, 64, norm=norm, gn_groups=gn_groups)
        self.up4 = UpLayer(64, 32, norm=norm, gn_groups=gn_groups)
        self.last_conv = nn.Conv2d(32, dimensions, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x1_up = self.up1(x4, x5)
        x2_up = self.up2(x3, x1_up)
        x3_up = self.up3(x2, x2_up)
        x4_up = self.up4(x1, x3_up)
        output = self.last_conv(x4_up)
        return output
    
    
class EarlyStopper():
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False