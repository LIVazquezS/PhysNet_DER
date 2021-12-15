import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import os
import sys
import numpy as np
import argparse
import logging
import string
import random
from torch_ema import ExponentialMovingAverage
# For Time measurement
from datetime import datetime
from time import time
# Neural Network importations
from Neural_Net_evid import PhysNet
# from neural_network.activation_fn import *
from Neural_Net_evid import gather_nd
from DataContainer import DataContainer
from utils import get_metric_func
data_tot = DataContainer('data/sn2_reactions.npz', 362167, 45000,
    100, 20, 0)
from tep import train
# assert torch.cuda.is_available()
# cuda_device = torch.device("cuda")
#%%

train_batches,N_train_batches = data_tot.get_train_batches()
epoch = torch.tensor(1,requires_grad=False,dtype=torch.int64)
#
model = PhysNet(device='cpu')

def get_indices(Nref,device='cpu'):
    # Get indices pointing to batch image
    # For some reason torch does not make repetition for float
    batch_seg = torch.arange(0, Nref.size()[0],device=device).repeat_interleave(Nref.type(torch.int64))
    # Initiate auxiliary parameter
    Nref_tot = torch.tensor(0, dtype=torch.int32).to(device)

    # Indices pointing to atom at each batch image
    idx = torch.arange(end=Nref[0], dtype=torch.int32).to(device)
    # Indices for atom pairs ij - Atom i
    Ntmp = Nref.cpu()
    idx_i = idx.repeat(int(Ntmp.numpy()[0]) - 1) + Nref_tot
    # Indices for atom pairs ij - Atom j
    idx_j = torch.roll(idx, -1, dims=0) + Nref_tot
    for Na in torch.arange(2, Nref[0]):
        Na_tmp = Na.cpu()
        idx_j = torch.concat(
            [idx_j, torch.roll(idx, int(-Na_tmp.numpy()), dims=0) + Nref_tot],
            dim=0)

    # Increment auxiliary parameter
    Nref_tot = Nref_tot + Nref[0]

    # Complete indices arrays
    for Nref_a in Nref[1:]:

        rng_a = torch.arange(end=Nref_a).to(device)
        Nref_a_tmp = Nref_a.cpu()
        idx = torch.concat(
            [idx, rng_a], axis=0)
        idx_i = torch.concat(
            [idx_i, rng_a.repeat(int(Nref_a_tmp.numpy()) - 1) + Nref_tot],
            dim=0)
        for Na in torch.arange(1, Nref_a):
            Na_tmp = Na.cpu()
            idx_j = torch.concat(
                [idx_j, torch.roll(rng_a, int(-Na_tmp.numpy()), dims=0) + Nref_tot],
                dim=0)

        # Increment auxiliary parameter
        Nref_tot = Nref_tot + Nref_a

    # Combine indices for batch image and respective atoms
    idx = torch.stack([batch_seg, idx], dim=1)
    return idx, idx_i, idx_j, batch_seg

def calculate_interatomic_distances(R, idx_i, idx_j, offsets=None):
    ''' Calculate interatomic distances '''

    Ri = torch.gather(R, 0, idx_i.type(torch.int64).view(-1, 1).repeat(1, 3))
    Rj = torch.gather(R, 0, idx_j.type(torch.int64).view(-1, 1).repeat(1, 3))
    if offsets is not None:
        Rj = Rj + offsets
    p = nn.ReLU(inplace=True)
    m = p(torch.sum((Ri - Rj) ** 2, dim=-1))
    Dij = torch.sqrt(m)
    # ReLU: y = max(0, x), prevent negative sqrt
    return Dij

def evidential_layer(means, loglambdas, logalphas, logbetas):
    min_val = 1e-6
    lambdas = torch.nn.Softplus()(loglambdas) + min_val
    alphas = torch.nn.Softplus()(logalphas) + min_val + 1  # add 1 for numerical contraints of Gamma function
    betas = torch.nn.Softplus()(logbetas) + min_val

    # Return these parameters as the output of the model
    output = torch.stack((means, lambdas, alphas, betas),
                         dim=1)
    return output

def evidential(out):

    # Split the outputs into the four distribution parameters for energy and charge
    # means, loglambdas, logalphas, logbetas = torch.split(out,out.shape[1]//4,dim=1)
    means, loglambdas, logalphas, logbetas = out
    out_E = evidential_layer(means, loglambdas, logalphas, logbetas)
    return out_E

def evidential_loss(mu, v, alpha, beta, targets, lam=1, epsilon=1e-4):
    """
    Use Deep Evidential Regression negative log likelihood loss + evidential
        regularizer
    We will use the new version on the paper..
    :mu: pred mean parameter for NIG
    :v: pred lam parameter for NIG
    :alpha: predicted parameter for NIG
    :beta: Predicted parmaeter for NIG
    :targets: Outputs to predict
    :return: Loss
    """
    # Calculate NLL loss
    twoBlambda = 2*beta*(1+v)
    nll = 0.5*torch.log(np.pi/v) \
        - alpha*torch.log(twoBlambda) \
        + (alpha+0.5) * torch.log(v*(targets-mu)**2 + twoBlambda) \
        + torch.lgamma(alpha) \
        - torch.lgamma(alpha+0.5)

    L_NLL = nll #torch.mean(nll, dim=-1)

    # Calculate regularizer based on absolute error of prediction
    error = torch.abs((targets - mu))
    reg = error * (2 * v + alpha)
    L_REG = reg #torch.mean(reg, dim=-1)

    # Loss = L_NLL + L_REG
    # TODO If we want to optimize the dual- of the objective use the line below:
    loss = L_NLL + lam * (L_REG - epsilon)

    return loss

# def train(batch):
#     model.train()
#     N_t, Z_t, R_t, Eref_t, Earef_t, Fref_t, Qref_t, Qaref_t, Dref_t = batch
#     #     # Get indices
#     idx_t, idx_i_t, idx_j_t, batch_seg_t = get_indices(N_t)
#
#     # Gather data
#     Z_t = gather_nd(Z_t, idx_t)
#     R_t = gather_nd(R_t, idx_t)
#
#     if torch.count_nonzero(Earef_t) != 0:
#         Earef_t = gather_nd(Earef_t, idx_t)
#     if torch.count_nonzero(Fref_t) != 0:
#         Fref_t = gather_nd(Fref_t, idx_t)
#     if torch.count_nonzero(Qaref_t) != 0:
#         Qaref_t = gather_nd(Qaref_t, idx_t)
#     model.zero_grad()
#     out = model.energy(Z_t, R_t, idx_i_t, idx_j_t, Qref_t, batch_seg_t)
#
#     p = evidential(out)
#     # TODO  Here the view should be change to consider the size of the batch
#     loss = evidential_loss(p[:, 0], p[:, 1], p[:, 2], p[:, 3], Eref_t).view(100, 1)
#     loss = loss.sum() / len(Z_t)
#     loss.backward(retain_graph=True)
#     # #Gradient clip
#     nn.utils.clip_grad_norm_(model.parameters(), 1000)
#     return loss

def predict(batch):
    model.eval()
    N_v, Z_v, R_v, Eref_v, Earef_v, Fref_v, Qref_v, Qaref_v, Dref_v = batch
    # Get indices
    idx_v, idx_i_v, idx_j_v, batch_seg_v = get_indices(N_v)
    Z_v = gather_nd(Z_v, idx_v)
    R_v = gather_nd(R_v, idx_v)

    if torch.count_nonzero(Earef_v) != 0:
        Earef_v = gather_nd(Earef_v, idx_v)
    if torch.count_nonzero(Fref_v) != 0:
        Fref_v = gather_nd(Fref_v, idx_v)
    if torch.count_nonzero(Qaref_v) != 0:
        Qaref_v = gather_nd(Qaref_v, idx_v)

    # Gather data
    with torch.no_grad():
        out = model.energy(Z_v, R_v, idx_i_v, idx_j_v, Qref_v, batch_seg_v)
        preds_b = evidential(out)

    preds = preds_b

    p = []
    c = []
    var = []
    for i in range(len(preds)):
        # Switching to chemprop implementation
        means = np.array([preds[i][j] for j in range(len(preds[i])) if j % 4 == 0])
        lambdas = np.array([preds[i][j] for j in range(len(preds[i])) if j % 4 == 1])
        alphas = np.array([preds[i][j] for j in range(len(preds[i])) if j % 4 == 2])
        betas = np.array([preds[i][j] for j in range(len(preds[i])) if j % 4 == 3])
    #     # means, lambdas, alphas, betas = np.split(np.array(preds[i]), 4)
        inverse_evidence = 1. / ((alphas - 1) * lambdas)

        p.append(means[0])
    #     # NOTE: inverse-evidence (ie. 1/evidence) is also a measure of
    #     # confidence. we can use this or the Var[X] defined by NIG.
        c.append(inverse_evidence[0])
        var.append(betas[0] * inverse_evidence[0])
        # c.append(betas / ((alphas-1) * lambdas))
    # These are BOTH variances
    # c = (scaler2 * c)
    # var = (scaler2 * var)
    Eref_v = Eref_v.detach().numpy()
    return p, c, var, Eref_v


#------------------------------------
# Evaluation
#------------------------------------

def evaluate_predictions(preds, targets, metric_func='rmse'):
    m_func = get_metric_func(metric_func)
    results = m_func(targets, preds)
    return results

def evaluate(data,metric_func='rmse'):
    preds,c,var,targets= predict(batch=data)

    results = evaluate_predictions(
        preds=preds,
        targets=targets,
        metric_func=metric_func)
    return results
#You have to reduce the learning rate....
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4,
                             weight_decay=0.1, amsgrad=True)

lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0)
valid_batches = data_tot.get_valid_batches()

# for ib,batch in enumerate(valid_batches):
#     x = evaluate(batch)
#     print(x)
# # Actual Train
for ib,batch in enumerate(train_batches):
    loss, mae_energy, pnorm, gnorm = train(model,batch)
    optimizer.step()
    print(loss,mae_energy,pnorm,gnorm)





