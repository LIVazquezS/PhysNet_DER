import torch
import torch.nn as nn
import torch.nn.functional as F
from .activation_fn import ActivationFN
from .DenseLayer import DenseLayer
from .ResidualLayer import ResidualLayer
from .InteractionLayer import InteractionLayer

class InteractionBlock(nn.Module):

    def __init__(self, F, K, num_residual_atomic, num_residual_interaction,
                 activation_fn=None,rate=0.0,device='cpu'):
        super(InteractionBlock, self).__init__()
        self.device = device

        # Interaction Layer

        self.interaction = InteractionLayer(F,K,num_residual_interaction,
                                            activation_fn=activation_fn,rate=rate,
                                            device=self.device)

        # Residual Layers
        self.residual_layers = nn.ModuleList([ResidualLayer(F,F,activation_fn=activation_fn,rate=rate,device=self.device)
              for _ in range(num_residual_atomic)])

            #nn.Sequential(
           # *[ResidualLayer(F,F,activation_fn=activation_fn,rate=rate,device=self.device)
           #   for _ in range(num_residual_atomic)])


    def forward(self,x,rbf,idx_i,idx_j):
        x = self.interaction(x,rbf,idx_i,idx_j)
        for i in range(len(self.residual_layers)):
             x = self.residual_layers[i](x)
        return x