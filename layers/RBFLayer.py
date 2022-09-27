import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
from .DenseLayer import *


def softplus_inverse(x):
    ''' Inverse softplus transformation ''' 
    return x + np.log(-np.expm1(-x))

class RBFLayer(nn.Module):
    ''' Radial basis function expansion '''

    def __init__(self, K, cutoff, dtype=torch.float32, device='cpu'):
        
        super(RBFLayer, self).__init__()
        self.device = device
        self.K = K
        self.cutoff = cutoff
        # Initialize centers (inverse softplus transformation is applied, 
        # such that softplus can be used to guarantee positive values)
        centers = softplus_inverse(np.linspace(1.0,np.exp(-self.cutoff),K))
        softp = nn.Softplus()
        self.centers = softp(nn.Parameter(torch.tensor(np.asarray(centers),dtype=dtype,device=self.device)))
		# Initialize widths (inverse softplus transformation is applied, 
        # such that softplus can be used to guarantee positive values)
        widths = [softplus_inverse((0.5/((1.0-np.exp(-self.cutoff))/K))**2)]*K
        self.widths = softp(nn.Parameter(torch.tensor(np.asarray(widths),dtype=dtype,device=self.device)))

		
    def cutoff_fn(self, D):
        ''' Cutoff function that ensures a smooth cutoff '''
        
        x = D/self.cutoff
        x3 = x**3
        x4 = x3*x
        x5 = x4*x
        return torch.where(x < 1, 1 - 6*x5 + 15*x4 - 10*x3, torch.zeros_like(x))
    
    def forward(self, D):
        
        D = torch.unsqueeze(D,-1) #necessary for proper broadcasting behaviour
        rbf = self.cutoff_fn(D)*(torch.exp(-self.widths*(torch.exp(-D) - self.centers)**2))
        
        return rbf







