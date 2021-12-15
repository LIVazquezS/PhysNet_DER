import torch
import torch.nn as nn
from .DenseLayer import DenseLayer
from .ResidualLayer import ResidualLayer
from .activation_fn import ActivationFN

class OutputBlock(nn.Module):
    
    def __init__(self,F, num_residual, activation_fn=None,rate=0.0,n_output=1,device='cpu'):
        #This has to be the number of outputs by the two because it returns one value for energy and one for charge
        super(OutputBlock,self).__init__()
        self.device = device
        self.activation_fn = activation_fn
        self.residual_layer = nn.Sequential(
            *[ResidualLayer(F,F,activation_fn=activation_fn,rate=rate,device=self.device) for _ in range(num_residual)]
        )
        #This has to be the number of outputs by the two because it returns one value for energy and one for charge
        self.n_output = 2 * n_output
        self.dense = DenseLayer(F,self.n_output,device=self.device)

    def forward(self,x):
        x1 = self.residual_layer(x)
        if self.activation_fn is not None:
            m = ActivationFN()
            x1 = m(self.activation_fn, x1)

        x2 = self.dense(x1)
        return x2
