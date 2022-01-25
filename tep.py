import torch
import torch.nn as nn
import numpy as np
# Neural Network importations
from Neural_Net_evid import PhysNet
from utils import get_metric_func
# from neural_network.activation_fn import *
from Neural_Net_evid import gather_nd
#====================================
# Some functions
#====================================
def compute_pnorm(model):
    """Computes the norm of the parameters of a model."""
    return np.sqrt(sum([p.norm().item() ** 2 for p in model.parameters()]))


def compute_gnorm(model):
    """Computes the norm of the gradients of a model."""
    return np.sqrt(sum([p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None]))

#------------------------------------
# Helping functions
#------------------------------------
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
        idx = torch.concat([idx, rng_a], axis=0)
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
    #Reorder the idx in i
    idx_i = torch.sort(idx_i)[0]
    # Combine indices for batch image and respective atoms
    idx = torch.stack([batch_seg, idx], dim=1)
    return idx.type(torch.int64), idx_i.type(torch.int64), idx_j.type(torch.int64), batch_seg.type(torch.int64)

#------------------------------------
# Loss function
#------------------------------------

def evidential_loss_new(mu, v, alpha, beta, targets, lam=1, epsilon=1e-4):
    """
    I use 0.2 as the found it as the best value on their paper.
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

def gauss_loss(mu,sigma,targets):
    """
    This defines a simple loss function for learning the log-likelihood of a gaussian distribution.
    In the future, we should use a regularizer.
    """
    loss = 0.5*np.log(2*np.pi) + 0.5*torch.log(sigma**2) + ((targets-mu)**2)/(2*sigma**2)

    return loss

def evidential_loss(mu, v, alpha, beta, targets):
    """
    Use Deep Evidential Regression Sum of Squared Error loss
    :mu: Pred mean parameter for NIG
    :v: Pred lambda parameter for NIG
    :alpha: predicted parameter for NIG
    :beta: Predicted parmaeter for NIG
    :targets: Outputs to predict
    :return: Loss
    """

    # Calculate SOS
    # Calculate gamma terms in front
    def Gamma(x):
        return torch.exp(torch.lgamma(x))

    coeff_denom = 4 * Gamma(alpha) * v * torch.sqrt(beta)
    coeff_num = Gamma(alpha - 0.5)
    coeff = coeff_num / coeff_denom

    # Calculate target dependent loss
    second_term = 2 * beta * (1 + v)
    second_term += (2 * alpha - 1) * v * torch.pow((targets - mu), 2)
    L_SOS = coeff * second_term

    # Calculate regularizer
    L_REG = torch.pow((targets - mu), 2) * (2 * alpha + v)

    loss_val = L_SOS + L_REG

    return loss_val



#------------------------------------
# Train Step
#------------------------------------

def train(model,batch,num_t,device,maxnorm=1000):
    model.train()
    # lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=2,
    #                                                          min_lr=1e-4,verbose=True)
    batch = [i.to(device) for i in batch]
    N_t, Z_t, R_t, Eref_t, Earef_t, Fref_t, Qref_t, Qaref_t, Dref_t = batch

    # Get indices
    idx_t, idx_i_t, idx_j_t, batch_seg_t = get_indices(N_t,device=device)

    # Gather data
    Z_t = gather_nd(Z_t, idx_t)
    R_t = gather_nd(R_t, idx_t)

    if torch.count_nonzero(Earef_t) != 0:
        Earef_t = gather_nd(Earef_t, idx_t)
    if torch.count_nonzero(Fref_t) != 0:
        Fref_t = gather_nd(Fref_t, idx_t)
    if torch.count_nonzero(Qaref_t) != 0:
        Qaref_t = gather_nd(Qaref_t, idx_t)
    # model.zero_grad()
    out = model.energy_evidential(Z_t, R_t, idx_i_t, idx_j_t, Qref_t, batch_seg_t)
    mae_energy = torch.mean(torch.abs(out[0] - Eref_t))

    loss = evidential_loss_new(out[0], out[1], out[2], out[3], Eref_t).sum()
    loss.backward(retain_graph=True)
    # lr_schedule.step(loss)
    # #Gradient clip
    nn.utils.clip_grad_norm_(model.parameters(),maxnorm)
    pnorm = compute_pnorm(model)
    gnorm = compute_gnorm(model)
    num_t = num_t + N_t.dim()
    return num_t, loss, mae_energy, pnorm, gnorm

def gauss_train(model,batch,num_t,device,maxnorm=1000):
    model.train()
    # lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=2,
    #                                                          min_lr=1e-4,verbose=True)
    batch = [i.to(device) for i in batch]
    N_t, Z_t, R_t, Eref_t, Earef_t, Fref_t, Qref_t, Qaref_t, Dref_t = batch

    # Get indices
    idx_t, idx_i_t, idx_j_t, batch_seg_t = get_indices(N_t,device=device)

    # Gather data
    Z_t = gather_nd(Z_t, idx_t)
    R_t = gather_nd(R_t, idx_t)

    if torch.count_nonzero(Earef_t) != 0:
        Earef_t = gather_nd(Earef_t, idx_t)
    if torch.count_nonzero(Fref_t) != 0:
        Fref_t = gather_nd(Fref_t, idx_t)
    if torch.count_nonzero(Qaref_t) != 0:
        Qaref_t = gather_nd(Qaref_t, idx_t)
    model.zero_grad()
    out = model.energy_gauss(Z_t, R_t, idx_i_t, idx_j_t, Qref_t, batch_seg_t)
    mae_energy = torch.mean(torch.abs(out[0] - Eref_t))

    loss = gauss_loss(out[0], out[1], Eref_t).sum()
    loss.backward(retain_graph=True)
    # lr_schedule.step(loss)
    # #Gradient clip
    nn.utils.clip_grad_norm_(model.parameters(),maxnorm)
    pnorm = compute_pnorm(model)
    gnorm = compute_gnorm(model)
    num_t = num_t + N_t.dim()
    return num_t, loss, mae_energy, pnorm, gnorm


#====================================
# Prediction
#====================================
def predict(model,batch,num_v,device):
    model.eval()
    batch = [i.to(device) for i in batch]
    N_v, Z_v, R_v, Eref_v, Earef_v, Fref_v, Qref_v, Qaref_v, Dref_v = batch
    # Get indices
    idx_v, idx_i_v, idx_j_v, batch_seg_v = get_indices(N_v,device=device)
    Z_v = gather_nd(Z_v, idx_v)
    R_v = gather_nd(R_v, idx_v)

    if torch.count_nonzero(Earef_v) != 0:
        Earef_v = gather_nd(Earef_v, idx_v)
    if torch.count_nonzero(Fref_v) != 0:
        Fref_v = gather_nd(Fref_v, idx_v)
    if torch.count_nonzero(Qaref_v) != 0:
        Qaref_v = gather_nd(Qaref_v, idx_v)

    with torch.no_grad():
        out = model.energy_evidential(Z_v, R_v, idx_i_v, idx_j_v, Qref_v, batch_seg_v)

    # loss
    preds = out

    loss = evidential_loss(preds[0], preds[1], preds[2], preds[3], Eref_v).sum()
    # preds = preds.cpu().detach().numpy()
    p = preds[0]
    c = 1 /((preds[2]-1)*preds[1])
    var = preds[3]*c
    # Eref_v = Eref_v.detach().cpu().numpy()
    Error_v = torch.mean(torch.abs(p - Eref_v)).detach().numpy()
    num_v = num_v + N_v.dim()
    return num_v,loss, p, c, var, Error_v

def gauss_predict(model,batch,num_v,device):
    model.eval()
    batch = [i.to(device) for i in batch]
    N_v, Z_v, R_v, Eref_v, Earef_v, Fref_v, Qref_v, Qaref_v, Dref_v = batch
    # Get indices
    idx_v, idx_i_v, idx_j_v, batch_seg_v = get_indices(N_v,device=device)
    Z_v = gather_nd(Z_v, idx_v)
    R_v = gather_nd(R_v, idx_v)

    if torch.count_nonzero(Earef_v) != 0:
        Earef_v = gather_nd(Earef_v, idx_v)
    if torch.count_nonzero(Fref_v) != 0:
        Fref_v = gather_nd(Fref_v, idx_v)
    if torch.count_nonzero(Qaref_v) != 0:
        Qaref_v = gather_nd(Qaref_v, idx_v)

    with torch.no_grad():
        out = model.energy_gauss(Z_v, R_v, idx_i_v, idx_j_v, Qref_v, batch_seg_v)

    # loss
    preds = out

    loss = gauss_loss(preds[0], preds[1], Eref_v).sum()
    # preds = preds.cpu().detach().numpy()
    p = preds[0]
    s = preds[1]
    # c = 1 /((preds[2]-1)*preds[1])
    # var = preds[3]*c
    # Eref_v = Eref_v.detach().cpu().numpy()
    Error_v = torch.mean(torch.abs(p - Eref_v)).detach().numpy()
    num_v = num_v + N_v.dim()
    return num_v,loss, p, s, Error_v

#------------------------------------
# Evaluation
#------------------------------------

def evaluate_predictions(preds, targets,metric_func='rmse' ):
    m_func = get_metric_func(metric_func)
    results = m_func(targets, preds)
    return results

def evaluate(model,data,metric_func='rmse'):
    preds,c,var,targets= predict(model=model,batch=data)

    results = evaluate_predictions(
        preds=preds,
        targets=targets,
        metric_func=metric_func)
    return results

