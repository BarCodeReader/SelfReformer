from .Encoder_swin import Encoder
from .Transformer import Transformer
from .CRM import CRM
from .Fuser import Fuser
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self,opt):
        super(Net,self).__init__()
        
        self.encoder = Encoder()
        # global context
        self.encoder_tf_ss = Transformer(depth=2, 
                                 num_heads=1,
                                 embed_dim=256, 
                                 mlp_ratio=3, 
                                 num_patches=196)

        self.encoder_shaper_7 = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 1024), nn.GELU())
        self.encoder_shaper_14 = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 256), nn.GELU())
        self.encoder_shaper_28 = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 64), nn.GELU())
        self.encoder_shaper_56 = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 16), nn.GELU())
        
        self.encoder_merge7_14 = nn.Sequential(nn.BatchNorm2d(512),
                                       nn.Conv2d(512,256,kernel_size=3, padding=1, bias=True), 
                                       nn.LeakyReLU())
        self.encoder_merge28_14 = nn.Sequential(nn.BatchNorm2d(512),
                                       nn.Conv2d(512,256,kernel_size=3, padding=1, bias=True), 
                                       nn.LeakyReLU())
        self.encoder_merge56_14 = nn.Sequential(nn.BatchNorm2d(512),
                                       nn.Conv2d(512,256,kernel_size=3, padding=1, bias=True), 
                                       nn.LeakyReLU())
        
        self.encoder_pred = nn.Sequential(nn.LayerNorm(256),
                                  nn.Linear(256, 256),
                                  nn.GELU(),
                                  nn.LayerNorm(256),
                                  nn.Linear(256, 256),
                                  nn.GELU(),
                                  nn.LayerNorm(256),
                                  nn.Linear(256, 1)
                                 )
        # main network
        self.transformer = nn.ModuleList([Transformer(depth=d, 
                                                      num_heads=n,
                                                      embed_dim=e, 
                                                      mlp_ratio=m, 
                                                      num_patches=p) for d,n,e,m,p in opt.transformer])

        self.fuser7_14 = Fuser(emb_dim=512, hw=7, cur_stg=1024)
        self.fuser14_28 = Fuser(emb_dim=256, hw=14, cur_stg=512)
        self.fuser28_56 = Fuser(emb_dim=128, hw=28, cur_stg=256)
        
        self.CRM_7 = CRM(inc=1024, outc=1024, hw=7, embed_dim=1024, num_patches=49)
        self.CRM_14 = CRM(inc=512, outc=256, hw=14, embed_dim=512, num_patches=196)
        self.CRM_28 = CRM(inc=256, outc=64, hw=28, embed_dim=256, num_patches=784)
        self.CRM_56 = CRM(inc=128, outc=16, hw=56, embed_dim=128, num_patches=3136)
        
    def forward(self, x):
        B = x.shape[0]
        # PVT encoder
        out_7r, out_14r, out_28r, out_56r = self.encoder(x) # is a cat_feature, list in shape of 16, 32, 64, 128
        pred = list()
        
        # ----------------------------for self-sup
        # reshape
        out_7s = self.encoder_shaper_7(out_7r).transpose(1,2).reshape(B,1024,7,7)
        out_7s = F.pixel_shuffle(out_7s, 2)
        
        out_14s = self.encoder_shaper_14(out_14r).transpose(1,2).reshape(B,256,14,14)
        
        out_28s = self.encoder_shaper_28(out_28r).transpose(1,2).reshape(B,64,28,28)
        out_28s = F.pixel_unshuffle(out_28s, 2)
        
        out_56s = self.encoder_shaper_56(out_56r).transpose(1,2).reshape(B,16,56,56)
        out_56s = F.pixel_unshuffle(out_56s, 4)
        
        # merge
        out = self.encoder_merge7_14(torch.cat([out_14s, out_7s], dim=1))
        out = self.encoder_merge28_14(torch.cat([out, out_28s], dim=1))
        out = self.encoder_merge56_14(torch.cat([out, out_56s], dim=1))
        out = out.reshape(B,256,-1).transpose(1,2) # B,N,C
        
        # pred
        out = self.encoder_tf_ss(out)
        matt = self.encoder_pred(out).transpose(1,2).reshape(B,1,14,14)
        pred.append(matt)
        
        matt = matt.repeat(1,256,1,1)
        matt = F.pixel_shuffle(matt,16) # B,1,224,224
        
        # ----------------------------for SOD
        out_7, out_14, out_28, out_56 = [tf(o, peb) for tf, o, peb in zip(self.transformer, 
                                                                          [out_7r, out_14r, out_28r, out_56r], 
                                                                          [False, False, False, False])] # B, patch, feature

        # 7
        p1_7, p2_7, out_7 = self.CRM_7(out_7, matt)
        pred.append(p1_7)
        pred.append(p2_7)
        # 14
        out_14 = self.fuser7_14(out_14, out_7)
        p1_14, p2_14, out_14 = self.CRM_14(out_14, matt)
        pred.append(p1_14)
        pred.append(p2_14)
        
        # 28
        out_28 = self.fuser14_28(out_28, out_14)
        p1_28, p2_28, out_28 = self.CRM_28(out_28, matt)
        pred.append(p1_28)
        pred.append(p2_28)
        
        # 56
        out_56 = self.fuser28_56(out_56, out_28)
        p1_56, p2_56, out_56 = self.CRM_56(out_56, matt)
        pred.append(p1_56)
        pred.append(p2_56)

        return pred
