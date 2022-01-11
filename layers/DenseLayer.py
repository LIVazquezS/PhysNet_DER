import torch
import torch.nn as nn
from .activation_fn import ActivationFN

class DenseLayer(nn.Module):
    
    def __init__(self, n_in, n_out, W_init=True,bias=True, activation_fn=None,device='cpu'):
        super(DenseLayer, self).__init__()
        self.device=device
        self.activation_fn = activation_fn
        self.bias = bias
        self.linear = nn.Linear(n_in, n_out,bias=self.bias,device=self.device)

        if W_init == True:
            nn.init.xavier_normal_(self.linear.weight)
        else:
            nn.init.zeros_(self.linear.weight)

        if bias == True:
            nn.init.zeros_(self.linear.bias)

    def forward(self,x):
        if self.activation_fn is not None:
            m = ActivationFN().to(self.device)
            x1 = m(self.activation_fn,self.linear(x))
        else:
            x1 = self.linear(x)
        return x1

    
