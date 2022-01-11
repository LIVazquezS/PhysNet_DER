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
# Evidential layer
#------------------------------------

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
#------------------------------------
# Loss function
#------------------------------------

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

#------------------------------------
# Train Step
#------------------------------------

def train(model,batch,num_t,device,maxnorm=1000):
    model.train()
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
    out = model.energy_evidential(Z_t, R_t, idx_i_t, idx_j_t, Qref_t, batch_seg_t)

    p = evidential(out)
    mae_energy = torch.mean(torch.abs(p[:,0] - Eref_t))

    loss = evidential_loss(p[:, 0], p[:, 1], p[:, 2], p[:, 3], Eref_t).view(len(N_t), 1)
    loss = loss.sum() / len(N_t)
    loss.backward(retain_graph=True)
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

    # Gather data
    with torch.no_grad():
        out = model.energy_evidential(Z_v, R_v, idx_i_v, idx_j_v, Qref_v, batch_seg_v)
        preds_b = evidential(out)

    # loss
    preds = preds_b

    loss = evidential_loss(preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3], Eref_v).view(len(N_v), 1)
    loss = loss.sum() / len(N_v)

    preds = preds.cpu().detach().numpy()
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
    Error_v = np.mean(np.abs(p - Eref_v))
    num_v = num_v + N_v.dim()
    return num_v,loss, p, c, var, Error_v


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

