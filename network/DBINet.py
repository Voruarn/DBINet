import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import *
from .init_weights import init_weights
from .DBINet_Backbone import DBIBkb
   

class DBINet(nn.Module):
    def __init__(self, **kwargs):
        super(DBINet, self).__init__()      

        self.backbone  = DBIBkb()
        mid_ch=64
        eout_channels=[64, 256, 512, 1024, 2048]
        out_ch=1
      
        self.eside1=Conv(eout_channels[1], mid_ch)
        self.eside2=Conv(eout_channels[2], mid_ch)
        self.eside3=Conv(eout_channels[3], mid_ch)
        self.eside4=Conv(eout_channels[4], mid_ch)

        self.convertor=Convertor(mid_ch, mid_ch)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Decoder
        self.decoder4 = Decoder(mid_ch, mid_ch)
        self.decoder3 = Decoder(mid_ch, mid_ch)
        self.decoder2 = Decoder(mid_ch, mid_ch)
        self.decoder1 = Decoder(mid_ch, mid_ch)
        
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
        
        # channel squeeze
        c1=self.eside1(c1)
        c2=self.eside2(c2)
        c3=self.eside3(c3)
        c4=self.eside4(c4)

        # convertor
        ca1,ca2,ca3,ca4=self.convertor(c1,c2,c3,c4)
        # Feedback from convertor
        ca12=F.interpolate(ca1, size=ca2.size()[2:], mode='bilinear', align_corners=True)
        ca13=F.interpolate(ca1, size=ca3.size()[2:], mode='bilinear', align_corners=True)
        ca14=F.interpolate(ca1, size=ca4.size()[2:], mode='bilinear', align_corners=True)

        # decoder
        up4=self.decoder4(ca14 + c4 + c5)
        up3=self.decoder3(ca13 + c3 + self.upsample2(up4))
        up2=self.decoder2(ca12 + c2 + self.upsample2(up3))
        up1=self.decoder1(ca1 + c1 + self.upsample2(up2))
        
        # side output saliency maps
        d1=self.dside1(up1)
        d2=self.dside2(up2)
        d3=self.dside3(up3)
        d4=self.dside4(up4)

        S1 = F.interpolate(d1, size=(H, W), mode='bilinear', align_corners=True)
        S2 = F.interpolate(d2, size=(H, W), mode='bilinear', align_corners=True)
        S3 = F.interpolate(d3, size=(H, W), mode='bilinear', align_corners=True)
        S4 = F.interpolate(d4, size=(H, W), mode='bilinear', align_corners=True)
     
  
        return S1,S2,S3,S4, torch.sigmoid(S1), torch.sigmoid(S2), torch.sigmoid(S3), torch.sigmoid(S4)





