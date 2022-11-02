import torch
import torch.nn as nn
import torch.nn.functional as F
from .activation_fn import *
from .DenseLayer import DenseLayer
from .ResidualLayer import ResidualLayer
from .utils import segment_sum

class InteractionLayer(nn.Module):

    def __init__(self, K, F, num_residual, activation_fn=None, rate=0.0,device='cpu'):
        super(InteractionLayer, self).__init__()
        #Device
        self.device = device
        # Droupout rate
        self.rate = rate
        # Dropout layer
        self.drop = nn.Dropout(self.rate)
        # Activation function
        self.activation_fn = activation_fn
        # Dense layers
        self.k2f = DenseLayer(K, F,W_init=False,bias=False, device=self.device)
        self.dense_i = DenseLayer(F, F, activation_fn=self.activation_fn,device=self.device)
        self.dense_j = DenseLayer(F, F, activation_fn=self.activation_fn,device=self.device)

        # Residual layers
        self.residual_layer = nn.ModuleList([ResidualLayer(F, F, activation_fn=self.activation_fn,
                                                           rate=self.rate,device=self.device) for _ in range(num_residual)])
        # For performing the final update to the feature vectors
        self.dense = DenseLayer(F, F,device=self.device)
        self.u = nn.Parameter(torch.ones([F], device=self.device,dtype=torch.float32))

        # torch.histogram(self.u)

    def forward(self, x, rbf, idx_i, idx_j):
        # Pre-activation
        if self.activation_fn is not None:
            xa = self.drop(self.activation_fn(x))
        else:
            xa = self.drop(x)
        # Calculate feature mask from radial basis functions
        g = self.k2f(rbf)
        # calculate contribution of neighbors and central atom
        xi = self.dense_i(xa)
        j = idx_j.view(-1, 1).expand(-1, x.shape[-1]).type(torch.int64)
        pxj = g * torch.gather(self.dense_j(xa), 0,j)
        xj = segment_sum(pxj, idx_i.type(torch.int64), device=self.device)

        # Do the sum of messages
        message = xi + xj
        # Residual layers
        for i in range(len(self.residual_layer)):
            message = self.residual_layer[i](message)
        if self.activation_fn is not None:
            message = self.activation_fn(message)

        x = self.u*x + self.dense(message)

        return x
