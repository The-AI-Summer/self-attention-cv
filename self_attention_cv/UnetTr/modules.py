import torch.nn as nn


# yellow block in Fig.1
class Conv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, double=True):
        super().__init__()
        if double:
            self.conv_block = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1,
                          padding=((kernel_size - 1) // 2)),
                nn.BatchNorm3d(out_planes),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_planes, out_planes, kernel_size=kernel_size, stride=1,
                          padding=((kernel_size - 1) // 2)),
                nn.BatchNorm3d(out_planes),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv_block = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1,
                          padding=((kernel_size - 1) // 2)),
                nn.BatchNorm3d(out_planes),
                nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv_block(x)

# green block in Fig.1
class TranspConv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=2, stride=2, padding=0, output_padding=0)

    def forward(self, x):
        return self.block(x)

# blue box in Fig.1
class BlueBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.Sequential(TranspConv3DBlock(in_planes, out_planes),
                                    Conv3DBlock(out_planes, out_planes,double=False))
    def forward(self, x):
        return self.block(x)