import torch.nn as nn
from scipy.stats import pearsonr
import torch

class Neg_Pearson_Loss(nn.Module):   
    #Taken from https://stackoverflow.com/a/19710598/11170350
    def __init__(self):
        super(Neg_Pearson_Loss,self).__init__()
        return
    def forward(self, X, Y):       
        assert not torch.any(torch.isnan(X))
        assert not torch.any(torch.isnan(Y))
        # Normalise X and Y
        X = X-X.mean(1)[:, None]
        Y = Y- Y.mean(1)[:, None]
        # Standardise X and Y
        X = (X/ X.std(1)[:, None])+1e-5
        Y =(Y/ Y.std(1)[:, None])+1e-5
        #multiply X and Y
        Z=(X*Y).mean(1)
        Z=1-Z.mean()
        return Z

def get_loss_function(name):
    if name == 'l2':
        return nn.MSELoss()
    elif name == 'plcc':
        return Neg_Pearson_Loss()
    else:
        raise ValueError("Invalid loss function name")
