import torch.nn as nn
from scipy.stats import pearsonr

class Neg_Pearson_Loss(nn.Module):
    def __init__(self):
        super(Neg_Pearson_Loss, self).__init__()

    def forward(self, x, y):
        return -pearsonr(x.detach().cpu().numpy(), y.detach().cpu().numpy())[0]

def get_loss_function(name):
    if name == 'l2':
        return nn.MSELoss()
    elif name == 'plcc':
        return Neg_Pearson_Loss()
    else:
        raise ValueError("Invalid loss function name")
