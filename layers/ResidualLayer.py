import torch
import torch.nn as nn
import torch.nn.functional as F
from .DenseLayer import DenseLayer
from .activation_fn import ActivationFN

class ResidualLayer(nn.Module):
    
    def __init__(self,n_in, n_out, activation_fn=None,rate=0.0,device='cpu'):
        super(ResidualLayer, self).__init__()
        self.device = device
        self.residual = DenseLayer(n_in, n_out, activation_fn=None,device=self.device)
        self.dense = DenseLayer(n_in, n_out, activation_fn=activation_fn,device=self.device)
        self.dropout = nn.Dropout(p=rate)
        self.activation_fn = activation_fn

    def forward(self,x):
        #Preactivation
        if self.activation_fn is not None:
            m = ActivationFN()
            y = self.dropout(m(self.activation_fn,x))
        else:
            y = self.dropout(x)
        #Residual
        x = x + self.residual(self.dense(y))
        return x


