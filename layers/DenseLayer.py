import torch
import torch.nn as nn
from .activation_fn import ActivationFN

class DenseLayer(nn.Module):
    
    def __init__(self, n_in, n_out, activation_fn=None,device='cpu'):
        super(DenseLayer, self).__init__()
        self.device=device
        self.activation_fn = activation_fn
        self.linear = nn.Linear(n_in, n_out,device=self.device)

    def forward(self,x):
        if self.activation_fn is not None:
            m = ActivationFN().to(self.device)
            x1 = m(self.activation_fn,self.linear(x))
        else:
            x1 = self.linear(x)
        return x1

    
