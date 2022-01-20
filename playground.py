#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 17:52:11 2021

@author: vazquez
"""
import torch
import torch.utils.data as data 
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
from DataContainer import DataContainer
from NeuralNet import PhysNet
from layers.RBFLayer import RBFLayer
from layers.InteractionBlock import InteractionBlock
from layers.OutputBlock import OutputBlock
from NeuralNet import gather_nd
from layers.utils import segment_sum

data_tot = DataContainer('data/qm9_1000.npz',800, 100,
    10, 20, 0)

# assert torch.cuda.is_available()
# cuda_device = torch.device("cuda")
#%%

train_batches,N_train_batches = data_tot.get_train_batches()
epoch = torch.tensor(1,requires_grad=False,dtype=torch.int64) 
#
# model = PhysNet()


# model.to(cuda_device)
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
#
F=128
K=64
sr_cut=10.0
num_blocks=5
num_residual_atomic=2
num_residual_interaction=3
num_residual_output=1
activation_fn="shift_softplus"
rate = 0.0
dtype=torch.float32
Eshift = 0.0
# Initial value for output energy scale
# (makes convergence faster)
Escale = 1.0
# Initial value for output charge shift
Qshift = 0.0
# Initial value for output charge scale
Qscale = 1.0
rbf_layer = RBFLayer(K, sr_cut)

interaction_block = nn.ModuleList([InteractionBlock(
    F, K, num_residual_atomic, num_residual_interaction,
    activation_fn=activation_fn, rate=rate)
    for _ in range(num_blocks)])

output_block = nn.ModuleList([OutputBlock(
            F, num_residual_output, activation_fn=activation_fn, rate=rate)
            for _ in range(num_blocks)])

output_block_evid = nn.ModuleList([OutputBlock(
    F, num_residual_output,n_output=3, activation_fn=activation_fn,rate=rate)
    for _ in range(num_blocks)])


embeddings = torch.Tensor(95, F).uniform_(-np.sqrt(3), np.sqrt(3)).requires_grad_(True)

def evidential_layer(means, loglambdas, logalphas, logbetas):
    min_val = 1e-6
    lambdas = torch.nn.Softplus()(loglambdas) + min_val
    alphas = torch.nn.Softplus()(logalphas) + min_val + 1  # add 1 for numerical contraints of Gamma function
    betas = torch.nn.Softplus()(logbetas) + min_val

    # Return these parameters as the output of the model
    output = torch.stack((means, lambdas, alphas, betas),
                         dim=2).view(100,4)
    return output

def evidential(out):

    # Split the outputs into the four distribution parameters for energy and charge
    means, loglambdas, logalphas, logbetas = torch.split(out,out.shape[1]//4,dim=1)

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

    L_NLL = torch.mean(nll,dim=-1) #torch.mean(nll, dim=-1)

    # Calculate regularizer based on absolute error of prediction
    error = torch.abs((targets - mu))
    reg = error * (2 * v + alpha)
    L_REG = torch.mean(reg,dim=-1) #torch.mean(reg, dim=-1)

    # Loss = L_NLL + L_REG
    # TODO If we want to optimize the dual- of the objective use the line below:
    loss = L_NLL + lam * (L_REG - epsilon)

    return loss

Eshift = torch.empty(95,).new_full((95,), Eshift).type(dtype)
Escale = torch.empty(95,).new_full((95,), Escale).type(dtype)
Qshift = torch.empty(95,).new_full((95,), Qshift).type(dtype)
Qscale = torch.empty(95,).new_full((95,), Qscale).type(dtype)

optimizer = torch.optim.Adam(params=interaction_block.parameters(), lr=0.1,
                             weight_decay=0.1, amsgrad=True)

lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.9)
# # Actual Train
for ib,batch in enumerate(train_batches):
    N_t, Z_t, R_t, Eref_t, Earef_t, Fref_t, Qref_t, Qaref_t, Dref_t = batch
#     # Get indices
    idx_t, idx_i_t, idx_j_t, batch_seg_t = get_indices(N_t)
    out = model.energy_evidential(Z_t, R_t, idx_i_t, idx_j_t, Qref_t, batch_seg_t)
    p = evidential(out)
    print(p)

 # Gather data
 #    Z_t = gather_nd(Z_t, idx_t)
 #    R_t = gather_nd(R_t, idx_t)
 #
 #    if torch.count_nonzero(Earef_t) != 0:
 #        Earef_t = gather_nd(Earef_t, idx_t)
 #    if torch.count_nonzero(Fref_t) != 0:
 #        Fref_t = gather_nd(Fref_t, idx_t)
 #    if torch.count_nonzero(Qaref_t) != 0:
 #        Qaref_t = gather_nd(Qaref_t, idx_t)
 #
 #    Dij_lr = calculate_interatomic_distances(R_t, idx_i_t, idx_j_t)
 #
 #
 #    # Calculate radial basis function expansion
 #    rbf = rbf_layer(Dij_lr)
 #
 #    # Initialize feature vectors according to embeddings for
 #    # nuclear charges
 #    z_pros = Z_t.view(-1, 1).expand(-1, 128).type(torch.int64)
 #    x = torch.gather(embeddings, 0, z_pros)
 #    # Apply blocks
 #    Ea = 0  # atomic energy
 #    Qa = 0  # atomic charge
 #    lambdas, alpha, beta = 0, 0, 0
 #    nhloss = 0  # non-hierarchicality loss
 #    for i in range(num_blocks):
 #        x = interaction_block[i](x, rbf, idx_i_t, idx_j_t)
 #        out = output_block[i](x)
 #        out_extra = output_block_evid[i](x)
 #        Ea = Ea + out[:, 0]
 #        Qa = Qa + out[:, 1]
 #        lambdas = lambdas + out_extra[:, 0]
 #        alpha = alpha + out_extra[:, 1]
 #        beta = beta + out_extra[:, 2]
 #
 #
 #    bs_u = len(torch.unique(batch_seg_t))
 #    Ea = Ea.new_zeros(bs_u).index_add(0, batch_seg_t, Ea)
 #    lambdas = lambdas.new_zeros(bs_u).index_add(0, batch_seg_t, lambdas)
 #    alpha = alpha.new_zeros(bs_u).index_add(0, batch_seg_t, alpha)
 #    beta = beta.new_zeros(bs_u).index_add(0, batch_seg_t, beta)
 #
 #    # Ea = torch.gather(Escale, 0, Z_t.type(torch.int64)) * Ea \
 #    #         + torch.gather(Eshift, 0, Z_t.type(torch.int64))
 #    # lambdas = torch.gather(Escale, 0, Z_t.type(torch.int64)) * lambdas
 #    # alpha = torch.gather(Escale, 0, Z_t.type(torch.int64)) * alpha
 #    # beta = torch.gather(Escale, 0, Z_t.type(torch.int64)) * beta
 #    out_E = torch.stack([Ea, lambdas, alpha, beta])
 #    E = evidential(out_E)
 #
 #    #
 #    # mask = torch.Tensor([x is not None for x in Eref_t])
 #    # # targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in batch])
 #    means = E[:, [j for j in range(len(E[0])) if j % 4 == 0]]
 #    lambdas = E[:, [j for j in range(len(E[0])) if j % 4 == 1]]
 #    alphas = E[:, [j for j in range(len(E[0])) if j % 4 == 2]]
 #    betas = E[:, [j for j in range(len(E[0])) if j % 4 == 3]]
 #    targets = Eref_t.view(100,1)
 #    print('means',means)
 #    print('lambdas',lambdas)
 #    print('alphas',alphas)
 #    print('betas',betas)
 #    print('targets',targets)
 #    # #
 #    # #
 #    # loss_sum = 0
 #    optimizer.zero_grad()
 #    loss = evidential_loss(means, lambdas, alphas, betas, Eref_t)
 #    loss = loss.sum()/len(Eref_t)
 #    #
 #    # #
 #    # # loss_sum += loss.item()
 #    loss.backward(retain_graph=True)
 #    optimizer.step()
 #
 #    print(loss)

        # #TODO: Find a better way to do this.

        # out_Q = torch.stack([out[:,1],out_extra[:,3],out_extra[:,4],out_extra[:,5]]).transpose(0,1)

        # E = evidential(out_E)
        # extra.append(out_E[:,1:4])
        # Ea += E[:,0]


#     e = torch.squeeze(segment_coo(Ea, index=batch_seg_t, reduce="sum"))
#     E1 = e.clone()
#     Eref_t1 = Eref_t.clone()
#     mse = nn.MSELoss()
#     energy_loss = mse(E1, Eref_t1)
#     print(energy_loss)
#     energy_loss.backward()
#     #
#
#     # # # Evaluate model
#     # # energy_t, forces_t, Ea_t, Qa_t, nhloss_t = \
#     # #     model.energy_and_forces_and_atomic_properties(
#     # #         Z_t, R_t, idx_i_t, idx_j_t, Qref_t, batch_seg_t)
#     # #
#     # # if torch.count_nonzero(Ea_t) != 0:
#     # #     Ea_t = gather_nd(Ea_t,idx_t)
#     # # else:
#     # #     Ea_t = torch.zeros(energy_t.shape)
#     # #
#     # # if torch.count_nonzero(Qa_t) != 0:
#     # #     Qa_t = gather_nd(Qa_t,idx_t)
#     # # else:
#     # #     Qa_t = torch.zeros(energy_t.shape)
#     #
#     #
#     # # p_array = [energy_t, forces_t, Ea_t, Qa_t]
#     # # p_array1 = [Eref_t, Fref_t,  Earef_t, Qaref_t]
#     # # print('from data:',Earef_t.shape)
#     # # print('from NN',Ea_t.shape)
#     # # # print(Earef_t)
#     # # names = ['energy','forces','ea','qa']
#     # # for i in range(len(p_array1)):
#     # #     if p_array1[i].shape != p_array[i].shape:
#     # #        print(names[i],p_array1[i].shape,p_array[i].shape)
#     #
#     # optimizer.zero_grad()
#     # mse = nn.MSELoss()
#     # # energy = model.energy(Z_t, R_t, idx_i_t, idx_j_t, Qref_t, batch_seg_t).type(torch.float32)
#     # # print('nn:',energy.type)
#     # # print('Ref',Eref_t.type)
#     # energy = torch.zeros(Eref_t.shape).requires_grad_(True)
#     # forces = torch.autograd.grad([energy.sum()],[R_t.requires_grad_(True)],allow_unused=True)[0]
#     #
#     # # Now the total loss has two parts, energy loss and force loss
#     # energy_loss = mse(energy, Eref_t)
#     # force_loss = (mse(Fref_t, forces).sum(dim=(1, 2))).mean()
#     # loss_t = energy_loss + 52.91772105638412 * force_loss
#     # print(loss_t)
#     # loss_t.backward()
#     # # optimizer.step()
#     # # # loss = energy_loss + force_coefficient * force_loss
#     # # num_t, loss_avg_t, emse_avg_t, emae_avg_t, fmse_avg_t, fmae_avg_t, \
#     # # qmse_avg_t, qmae_avg_t, dmse_avg_t, dmae_avg_t = reset_averages()
#     # #
#     # # p = train_step(batch, num_t, loss_avg_t, emse_avg_t, emae_avg_t,
#     # #         fmse_avg_t, fmae_avg_t, qmse_avg_t, qmae_avg_t,
#     # #         dmse_avg_t, dmae_avg_t)
#     # #
#     # # print(p)
#     # optimizer.step()