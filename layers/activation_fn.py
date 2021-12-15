#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 12:13:09 2021

@author: vazquez
"""
import torch
import torch.nn as nn
import torch.nn.functional as f

#TODO: Rewritte this class

class ActivationFN(nn.Module):
    def __init__(self):
        super(ActivationFN, self).__init__()

    # Google's swish function
    def swish(self, x):
        return x * f.sigmoid(x)

    # First time softplus was used as activation function: "Incorporating
    # Second-Order Functional Knowledge for Better Option Pricing"
    # (https://papers.nips.cc/paper/1920-incorporating-second-order-functional-knowledge-for-better-option-pricing.pdf)
    def _softplus(self, x):
        return torch.log1p(torch.exp(x))

    def softplus(self, x):
        # This definition is for numerical stability for x larger than 15 (single
        # precision) or x larger than 34 (double precision), there is no numerical
        # difference anymore between the softplus and a linear function
        return torch.where(x < 15.0, self._softplus(torch.where(x < 15.0, x, torch.zeros_like(x))), x)

    def shifted_softplus(self, x):
        # return softplus(x) - np.log(2.0)
        return f.softplus(x) - torch.log(torch.tensor(2.0))

    # This ensures that the function is close to linear near the origin!
    def scaled_shifted_softplus(self, x):
        return 2 * self.shifted_softplus(x)

    # Is not really self-normalizing sadly...
    def self_normalizing_shifted_softplus(self, x):
        return 1.875596256135042 * self.shifted_softplus(x)

    # General: ln((exp(alpha)-1)*exp(x)+1)-alpha
    def smooth_ELU(self, x):  # check this
        return torch.log1p(1.718281828459045 * torch.exp(x)) - 1.0  # (e-1) = 1.718281828459045

    def self_normalizing_smooth_ELU(self, x):
        return 1.574030675714671 * self.smooth_ELU(x)

    def self_normalizing_asinh(self, x):
        return 1.256734802399369 * torch.asinh(x)

    def self_normalizing_tanh(self, x):
        return 1.592537419722831 * torch.tanh(x)

    def forward(self, name, x=None):
        if x is None:
            return
        self.name = name
        if self.name == 'softplus':
            return self.softplus(x)
        if self.name == 'shift_softplus':
            return self.shifted_softplus(x)
        if self.name == 'scaled_shift_softplus':
            return self.scaled_shifted_softplus(x)
        if self.name == 'snsf':
            return self.self_normalizing_shifted_softplus(x)
        if self.name == 'smooth_ELU':
            return self.smooth_ELU(x)
        if self.name == 'sn_smooth_ELU':
            return self.self_normalizing_smooth_ELU(x)
        if self.name == 'sn_asinh':
            return self.self_normalizing_asinh(x)
        if self.name == 'sn_tanh':
            return self.self_normalizing_tanh(x)
