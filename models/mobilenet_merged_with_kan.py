from efficient_kan import KAN
import torch
import torch.nn as nn
import timm

class MobileNetMergedWithKAN(nn.Module):
    def __init__(self, authentic_weights_path=None, synthetic_weights_path=None):
        super(MobileNetMergedWithKAN, self).__init__()
        
        self.authentic = timm.create_model("mobilenetv3_large_100.ra_in1k", pretrained=False)
        self.syntetic  = timm.create_model("mobilenetv3_large_100.ra_in1k", pretrained=False)
        
        if authentic_weights_path:
            self.authentic.load_state_dict(torch.load(authentic_weights_path, map_location=torch.device('cpu')))
        
        if synthetic_weights_path:
            self.syntetic.load_state_dict(torch.load(synthetic_weights_path, map_location=torch.device('cpu')))
        
        for param in self.authentic.parameters():
            param.requires_grad = True
        for param in self.syntetic.parameters():
            param.requires_grad = True

        self.aut_dw = KAN([1000, 512])
        self.syn_dw = KAN([1000, 512])

        self.head = KAN([1024, 1])

    def forward(self, inp, inp2):

        authentic = self.authentic(inp)
        syntetic = self.syntetic(inp2)
        
        authentic = self.aut_dw(authentic)
        syntetic = self.syn_dw(syntetic)

        concat_pool = torch.cat([authentic, syntetic], dim=1)
        head = self.head(concat_pool)

        return head

