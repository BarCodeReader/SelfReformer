import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .PVT_V2 import pvt_v2_b2


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()

        self.encoder = pvt_v2_b2()
        self.encoder.load_state_dict(torch.load('./model/pretrain/pvt_v2_b2.pth', map_location='cpu'), strict=False)

    def forward(self, x):
        out = self.encoder(x)
        return out[::-1] 