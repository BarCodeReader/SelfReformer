import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .SWIN import SwinB
from einops import rearrange


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()

        self.encoder = SwinB()
        self.encoder.load_state_dict(torch.load('./model/pretrain/swin_base_patch4_window12_384_22kto1k.pth', 
                                                map_location='cpu')['model'], 
                                     strict=False)

    def forward(self, x):
        _, out56, out28, out14, out7 = self.encoder(x)

        out56 = rearrange(out56, 'b c h w -> b (h w) c', h=56, w=56) # c=128
        out28 = rearrange(out28, 'b c h w -> b (h w) c', h=28, w=28) # c=256
        out14 = rearrange(out14, 'b c h w -> b (h w) c', h=14, w=14) # c=512
        out7  = rearrange(out7,  'b c h w -> b (h w) c', h=7, w=7)   # c=1024
        return out7, out14, out28, out56
