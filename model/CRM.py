import torch
import torch.nn as nn
from .Transformer import Transformer
import torch.nn.functional as F


class CRM(nn.Module):
    def __init__(self, inc, outc, hw, embed_dim, num_patches, depth=4):
        super(CRM, self).__init__()
        self.conv_p1 = nn.Conv2d(inc, outc, kernel_size=3, padding=1, bias=True)
        self.conv_p2 = nn.Conv2d(inc, outc, kernel_size=3, padding=1, bias=True)
        self.conv_glb = nn.Conv2d(outc, inc, kernel_size=3, padding=1, bias=True)

        self.conv_matt = nn.Sequential(nn.Conv2d(outc, inc, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(inc),
                                       nn.LeakyReLU(inplace=True))
        self.conv_fuse = nn.Sequential(nn.Conv2d(2 * inc, inc, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(inc),
                                       nn.LeakyReLU(inplace=True),
                                       nn.Conv2d(inc, inc, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(inc),
                                       nn.LeakyReLU(inplace=True),
                                       nn.Conv2d(inc, inc, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(inc),
                                       nn.LeakyReLU(inplace=True))

        self.sigmoid = nn.Sigmoid()
        self.tf = Transformer(depth=depth,
                              num_heads=1,
                              embed_dim=embed_dim,
                              mlp_ratio=3,
                              num_patches=num_patches)
        self.hw = hw
        self.inc = inc

    def forward(self, x, glbmap):
        # x in shape of B,N,C
        # glbmap in shape of B,1,224,224
        B, _, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, self.hw, self.hw)
        if glbmap.shape[-1] // self.hw != 1:
            glbmap = F.pixel_unshuffle(glbmap, glbmap.shape[-1] // self.hw)
            glbmap = self.conv_glb(glbmap)

        x = torch.cat([glbmap, x], dim=1)
        x = self.conv_fuse(x)
        # pred
        p1 = self.conv_p1(x)
        matt = self.sigmoid(p1)
        matt = matt * (1 - matt)
        matt = self.conv_matt(matt)
        fea = x * (1 + matt)

        # reshape x back to B,N,C
        fea = fea.reshape(B, self.inc, -1).transpose(1, 2)
        fea = self.tf(fea, True)
        p2 = self.conv_p2(fea.transpose(1, 2).reshape(B, C, self.hw, self.hw))

        return [p1, p2, fea]
