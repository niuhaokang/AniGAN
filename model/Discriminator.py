import torch.nn as nn
import numpy as np

class AniGANDiscriminator(nn.Module):
    def __init__(self, in_channel=3, ndf=64):
        super().__init__()
        self.avg = nn.AvgPool2d(3, stride=2)
        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        n_layers = 3
        share = [
            nn.Conv2d(in_channels=in_channel, out_channels=ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            share += [
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                nn.BatchNorm2d(nf),
                nn.LeakyReLU(0.2, True)
            ]
        self.share = nn.Sequential(*share)

        self.smooth = nn.Conv2d(in_channels=nf,
                                out_channels=1,
                                kernel_size=kw,
                                stride=1,
                                padding=padw)

        self.func_X = nn.Sequential(*[
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        ])

        self.func_Y = nn.Sequential(*[
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        ])

    def forward(self, z, domain):
        K1 = []
        K2 = []
        for i in range(len(self.share)):
            z = self.share[i](z)
            if i == 4 or i == 7:
                K1.append(z)

        x = self.smooth(z)
        y = self.smooth(z)

        if domain == 'X':
            for i in range(len(self.func_X)):
                x = self.func_X[i](x)
                if i == 0:
                    K2.append(x)
            return x, K1, K2
        elif domain == 'Y':
            for i in range(len(self.func_Y)):
                y = self.func_Y[i](y)
                if i == 0:
                    K2.append(y)
            return y, K1, K2
        else:
            print("模式不匹配")
            assert 1 > 2

if __name__ == '__main__':
    import torch
    m = AniGANDiscriminator()
    n = nn.AdaptiveAvgPool2d(1)
    a = torch.rand([4, 3, 256, 256])

    b,k1,k2 = m(a, 'X')
    print(b.size())
    print(k1[0].size(), k1[0].size())
    print(k2[0].size())