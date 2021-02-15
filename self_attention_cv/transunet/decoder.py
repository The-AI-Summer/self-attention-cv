import torch
import torch.nn as nn


class SignleConv(nn.Module):
    """
    Double convolution block that keeps that spatial sizes the same
    """

    def __init__(self, in_ch, out_ch, norm_layer=None):
        super(SignleConv, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            norm_layer(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class DoubleConv(nn.Module):
    """
    Double convolution block that keeps that spatial sizes the same
    """

    def __init__(self, in_ch, out_ch, norm_layer=None):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(SignleConv(in_ch, out_ch, norm_layer),
                                  SignleConv(out_ch, out_ch, norm_layer))

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    """
    Doubles spatial size with bilinear upsampling
    Skip connections and double convs
    """

    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        mode = "bilinear"
        self.up = nn.Upsample(scale_factor=2, mode=mode, align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2=None):
        """
        Args:
            x1: [b,c, h, w]
            x2: [b,c, 2*h,2*w]

        Returns: 2x upsampled double conv reselt
        """
        x = self.up(x1)
        if x2 is not None:
            x = torch.cat([x2, x], dim=1)
        return self.conv(x)
