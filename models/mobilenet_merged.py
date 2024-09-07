import torch
import torch.nn as nn
import timm

class MobileNetMerged(nn.Module):
    def __init__(self, block_size=4, authentic_weights_path=None, synthetic_weights_path=None):
        super(MobileNetMerged, self).__init__()

        self.authentic = timm.create_model("mobilenetv3_large_100.ra_in1k", pretrained=False)
        self.syntetic  = timm.create_model("mobilenetv3_large_100.ra_in1k", pretrained=False)
        
        if authentic_weights_path:
            self.authentic.load_state_dict(torch.load(authentic_weights_path, map_location=torch.device('cpu')))
        
        if synthetic_weights_path:
            self.syntetic.load_state_dict(torch.load(synthetic_weights_path, map_location=torch.device('cpu')))
        
        self.aut_up = nn.Linear(1000, 3100)
        self.syn_up = nn.Linear(1000, 3100)
        self.aut_dw = nn.Linear(3100, 512)
        self.syn_dw = nn.Linear(3100, 512)
        
        self.head = nn.Linear(1024, 1)
        self.relu = nn.ReLU()

    def forward(self, inp, inp2):

        authentic = self.authentic(inp)
        authentic = self.relu(authentic) 
        authentic = self.aut_up(authentic)
        authentic = self.relu(authentic)
        authentic = self.aut_dw(authentic)
        authentic = self.relu(authentic)

        syntetic = self.syntetic(inp2)
        syntetic = self.relu(syntetic)
        syntetic = self.syn_up(syntetic)
        syntetic = self.relu(syntetic)
        syntetic = self.syn_dw(syntetic)
        syntetic = self.relu(syntetic)

        concat_pool = torch.cat([authentic, syntetic], dim=1)
        output = self.head(concat_pool)

        return output
