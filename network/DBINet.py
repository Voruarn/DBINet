import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import *
from .init_weights import init_weights
from .DBINet_Backbone import DBIBkb


class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
   

class DBINet(nn.Module):
    def __init__(self, n_channels=3, is_deconv=True):
        super(DBINet, self).__init__()      

        self.backbone  = DBIBkb()
        mid_ch=64
        eout_channels=[64, 256, 512, 1024, 2048]
        out_ch=1
      
        self.is_deconv = is_deconv
        
        self.center1=CBAM(eout_channels[1], mid_ch)
        self.center2=CBAM(eout_channels[2], mid_ch)
        self.center3=CBAM(eout_channels[3], mid_ch)
        self.center4=CBAM(eout_channels[4], mid_ch)

        self.SP=SPPF(mid_ch, mid_ch)

        # Decoder
        self.decoder4 = Decoder(mid_ch, mid_ch, is_deconv=False)
        self.decoder3 = Decoder(mid_ch, mid_ch, self.is_deconv)
        self.decoder2 = Decoder(mid_ch, mid_ch, self.is_deconv)
        self.decoder1 = Decoder(mid_ch, mid_ch, self.is_deconv)
        
        self.dside1 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.dside2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.dside3 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.dside4 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, stride=1, padding=1)


        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)

        # encoder
        outs=self.backbone(inputs)
        c1, c2, c3, c4, c5 = outs

        c1=self.center1(c1)
        c2=self.center2(c2)
        c3=self.center3(c3)
        c4=self.center4(c4)
        c5=self.SP(c5)


        up4=self.decoder4(c5,c4)
        up3=self.decoder3(up4,c3)
        up2=self.decoder2(up3,c2)
        up1=self.decoder1(up2,c1)
        
        d1=self.dside1(up1)
        d2=self.dside2(up2)
        d3=self.dside3(up3)
        d4=self.dside4(up4)

        S1 = F.interpolate(d1, size=(H, W), mode='bilinear', align_corners=True)
        S2 = F.interpolate(d2, size=(H, W), mode='bilinear', align_corners=True)
        S3 = F.interpolate(d3, size=(H, W), mode='bilinear', align_corners=True)
        S4 = F.interpolate(d4, size=(H, W), mode='bilinear', align_corners=True)
  
        return torch.sigmoid(S1), torch.sigmoid(S2), torch.sigmoid(S3), torch.sigmoid(S4)





