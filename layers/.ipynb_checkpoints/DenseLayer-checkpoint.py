import torch
import torch.nn as nn
from .activation_fn import ActivationFN

class DenseLayer(nn.Module):
    
    def __init__(self, n_in, n_out, activation_fn=None):
        super(DenseLayer, self).__init__()
        self.activation_fn = activation_fn
        self.linear = nn.Linear(n_in, n_out)

    def forward(self,x, activation_fn=None):
        if activation_fn is not None:
            m = ActivationFN()
            x1 = m(self.linear(x))
        else:
            x1 = self.linear(x)
        return x1

    
