from efficient_kan import KAN
import torch
import torch.nn as nn
import timm

class MobileNetWithKAN(nn.Module):
    def __init__(self):
        super(MobileNetWithKAN, self).__init__()
        
        self.base = timm.create_model("mobilenetv3_large_100.ra_in1k", pretrained=False)

        self.aut_dw = KAN([1000, 128])
        
        self.head = KAN([128, 1])

    def forward(self, x):

        x = self.base(x)
        
        x = self.aut_dw(x)

        head = self.head(x)

        return head

