# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F
from network.model_parts import *
from thop import profile

class CModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CModel, self).__init__()
        self.inc = inconv(in_channels, 32)
        self.CAAM = CAAM(32)
        self.Mix1 = Mixer(32)
        self.down1 = down(32, 64)
        self.Mix2 = Mixer(64)
        self.down2 = down(64, 128)
        self.Mix3 = Mixer(128)
        self.down3 = down(128, 256)
        self.Mix4 = Mixer(256)
        self.down4 = down(256, 256)
        self.up1 = up(512, 128)
        self.up2 = up(256, 64)
        self.up3 = up(128, 32)
        self.up4 = up(64, 32)
        self.outc = outconv(32, out_channels)
        self.psp = PSPModule(256, 256, sizes=(1, 2, 3, 6))


    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x1 = self.inc(x)
        x1 = self.CAAM(x1)
        x1 = self.Mix1(x1)     # 1, 32, 384, 384
        x2 = self.down1(x1)
        x2 = self.Mix2(x2)     # 1, 64, 192, 192
        x3 = self.down2(x2)
        x3 = self.Mix3(x3)     # 1, 128, 96, 96
        x4 = self.down3(x3)
        x4 = self.Mix4(x4)     # 1, 256, 48, 48
        x5 = self.down4(x4)
        x5 = self.Mix4(x5)     # 1, 256, 24, 24
        x5 = self.psp(x5)
        x6 = self.up1(x5, x4)  # 1, 128, 48,  48   8
        x7 = self.up2(x6, x3)  # 1, 64,  96,  96   4
        x8 = self.up3(x7, x2)  # 1, 32,  192, 192  2
        x9 = self.up4(x8, x1)  # 1, 32,  384, 384
        x0 = self.outc(x9)

        return torch.sigmoid(x0)
    


if __name__ == '__main__':
    x = torch.rand(1, 384, 384, 3)
    model = CModel(3, 1)
    flops, params = profile(model, inputs=(x, ))
    print("FLOPs=", str(flops/1e9) +'{}'.format("G"))
    print("params=", str(params/1e6)+'{}'.format("M"))
    out1 = model(x)
    print(out1.shape)
