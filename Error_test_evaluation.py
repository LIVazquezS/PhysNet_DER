#!/usr/bin/env python3
import torch
import numpy as np
import os
import sys
import argparse
from Neural_Net_evid import PhysNet
from Neural_Net_evid import gather_nd
from layers.activation_fn import *
from DataContainer import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# define command line arguments
parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument("--restart", type=str, default='No',
                    help="Restart training from a specific folder")
parser.add_argument("--num_features", default=128, type=int)
parser.add_argument("--num_basis", default=64, type=int)
parser.add_argument("--num_blocks", default=5, type=int)
parser.add_argument("--num_residual_atomic", default=2, type=int)
parser.add_argument("--num_residual_interaction", default=3, type=int)
parser.add_argument("--num_residual_output", default=1, type=int)
parser.add_argument("--cutoff", default=10.0, type=float)
parser.add_argument("--use_electrostatic", default=1, type=int)
parser.add_argument("--use_dispersion", default=1, type=int)
parser.add_argument("--grimme_s6", default=None, type=float)
parser.add_argument("--grimme_s8", default=None, type=float)
parser.add_argument("--grimme_a1", default=None, type=float)
parser.add_argument("--grimme_a2", default=None, type=float)
parser.add_argument("--dataset", type=str)
parser.add_argument("--num_train", type=int)
parser.add_argument("--num_valid", type=int)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--valid_batch_size", type=int)
parser.add_argument("--seed", default=None, type=int)
parser.add_argument("--max_steps", default=10000, type=int)
parser.add_argument("--learning_rate", default=0.001, type=float)
parser.add_argument("--decay_steps", default=1000, type=int)
parser.add_argument("--decay_rate", default=0.1, type=float)
parser.add_argument("--max_norm", default=1000.0, type=float)
parser.add_argument("--ema_decay", default=0.999, type=float)
parser.add_argument("--rate", default=0.0, type=float)
parser.add_argument("--l2lambda", default=0.0, type=float)
parser.add_argument("--nhlambda", default=0.1, type=float)
#Note: This parameter is setup to 0.2 as it was the best value on the paper of Amini...
parser.add_argument("--lambda_conf", default=0.2, type=float,
                    help="Lambda value of the confidence of the prediction")
parser.add_argument("--summary_interval", default=5, type=int)
parser.add_argument("--validation_interval", default=5, type=int)
parser.add_argument("--show_progress", default=True, type=bool)
parser.add_argument("--save_interval", default=5, type=int)
parser.add_argument("--record_run_metadata", default=0, type=int)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--checkpoint', default='', type=str)

# if no command line arguments are present, config file is parsed
config_file = 'config.txt'
if len(sys.argv) == 1:
    if os.path.isfile(config_file):
        args = parser.parse_args(["@" + config_file])
    else:
        args = parser.parse_args(["--help"])
else:
    args = parser.parse_args()

model = PhysNet(
    F=args.num_features,
    K=args.num_basis,
    sr_cut=args.cutoff,
    num_blocks=args.num_blocks,
    num_residual_atomic=args.num_residual_atomic,
    num_residual_interaction=args.num_residual_interaction,
    num_residual_output=args.num_residual_output,
    use_electrostatic=(args.use_electrostatic == 1),
    use_dispersion=(args.use_dispersion == 1),
    s6=args.grimme_s6,
    s8=args.grimme_s8,
    a1=args.grimme_a1,
    a2=args.grimme_a2,
    writer=False,
    activation_fn=shifted_softplus,
    device=args.device)

# TODO: Add the option to load and evaluate multiple models as the same time
checkpoints = [args.checkpoint]
def load_checkpoint(checkpoints):
    if checkpoints[0] is not None:
        checkpoint = torch.load(checkpoints[0])
        return checkpoint


# Load neural network parameter
latest_ckpt = load_checkpoint(checkpoints)

model.load_state_dict(latest_ckpt['model_state_dict'])

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

def evid_test_step(batch,device):
    model.eval()
    batch = [i.to(device) for i in batch]
    N_test, Z_test, R_test, Eref_test, Earef_test, Fref_test, Qref_test, Qaref_test, Dref_test = batch
    # Get indices
    idx_test, idx_i_test, idx_j_test, batch_seg_test = get_indices(N_test,device=device)
    Z_test = gather_nd(Z_test, idx_test)
    R_test = gather_nd(R_test, idx_test)

    if torch.count_nonzero(Earef_test) != 0:
        Earef_test = gather_nd(Earef_test, idx_test)
    if torch.count_nonzero(Fref_test) != 0:
        Fref_test = gather_nd(Fref_test, idx_test)
    if torch.count_nonzero(Qaref_test) != 0:
        Qaref_test = gather_nd(Qaref_test, idx_test)

    with torch.no_grad():
        energy_test, lambdas_test, alpha_test, beta_test = \
            model.energy_evidential(Z_test, R_test, idx_i_test, idx_j_test, Qref_test, batch_seg=batch_seg_test)

    sigma2 = beta_test.detach().cpu().numpy() / (alpha_test.detach().cpu().numpy() - 1)
    var = (1 / lambdas_test.detach().cpu().numpy()) * sigma2
    energy_test_list = energy_test.detach().cpu().numpy()
    Eref_test_list = Eref_test.detach().cpu().numpy()
    # loss
    mae_energy = torch.mean(torch.abs(energy_test - Eref_test)).cpu().numpy()
    mse_energy = torch.mean(torch.square(energy_test - Eref_test)).cpu().numpy()
    rmse_energy = np.sqrt(mse_energy)
    return energy_test_list, Eref_test_list, mae_energy, mse_energy, rmse_energy, var,sigma2

def num_to_label(array):
    labels = []
    for j in array:
        l = j
        if l == 0:
            continue
        elif l == 1:
            labels.append('H')
        elif l == 6:
            labels.append('C')
        elif l == 7:
            labels.append('N')
        elif l == 8:
            labels.append('O')
        else:
            print("UNKNOWN ATOM", l)
            quit()
    return labels

#
def write(N_e,Z_e,R_e,E_e,id,folder,save=True):
    atom_template = '{:3s} \t {:15.10f} \t {:15.10f} \t {:15.10f} '
    for i,j in enumerate(N_e):
        block = []
        block.append(int(j))
        block.append(E_e[i])
        for element, (x, y, z) in zip(Z_e[i], R_e[i]):
            block.append(atom_template.format(element, x, y, z))

        if save:
            if os.path.exists(folder):
                os.chdir(folder)
            else:
                os.makedirs(folder)
                os.chdir(folder)

            with open('error_{}_{}.xyz'.format(id,i), 'w') as f:
                for item in block:
                    f.write("%s\n" % item)
            os.chdir('..')
        else:
            print(block)


def get_error_mols(folder,id,batch,var):
    perc = np.percentile(var, 95)
    indexes = np.argwhere(var > perc)
    batch = [i.detach().cpu().numpy() for i in batch]
    N, Z, R, E, _, _, _, _, _ = batch
    N_e = []
    Z_e = []
    R_e = []
    E_e = []
    for j in range(len(indexes)):
        value = indexes[j][0]
        N_e.append(N[value])
        Z_e.append(num_to_label(Z[value]))
        R_e.append(R[value])
        E_e.append(E[value])
    write(N_e,Z_e,R_e,E_e,id,folder)

# load dataset
data = DataContainer(
    args.dataset, args.num_train, args.num_valid,
    args.batch_size, args.valid_batch_size, seed=args.seed)

test_batches = data.get_test_batches()
print("Number of test batches:", len(test_batches))
eshift = data.EperA_m_n # The energy shift is the mean energy of the training dataset
escale = data.EperA_s_n # the energy scale is the standard deviation of the training dataset

MAE_by_batch = []
RMSE_by_batch = []
energies_by_batch = []
Eref_by_batch = []
var_by_batch = []
sigma2_by_batch = []
Error_by_mol_batch = []
for ib, batch in enumerate(test_batches):
    energy_test_list, Eref_test_list, mae_energy, mse_energy, rmse_energy, var,sigma2 = evid_test_step(batch,args.device)
    E_by_mol = np.abs(energy_test_list - Eref_test_list)
    get_error_mols('Error_1.0',ib,batch,var)
    Error_by_mol_batch.extend(E_by_mol)
    MAE_by_batch.append(mae_energy)
    RMSE_by_batch.append(rmse_energy)
    energies_by_batch.extend(energy_test_list)
    Eref_by_batch.extend(Eref_test_list)
    var_by_batch.extend(var)
    sigma2_by_batch.extend(sigma2)

# p = np.argwhere(np.array(var_by_batch) > 1.0)
#
# print(p)
MAE_final = np.mean(MAE_by_batch)
RMSE_final = np.mean(RMSE_by_batch)
print('MAE(kcal/mol): {:.4}'.format(MAE_final*23))
print('RMSE(kcal/mol): {:.4}'.format(RMSE_final*23))
#TODO: Make this as a flag
save = False
plot = False
save_plot = False


slope, intercept, r, p, se = stats.linregress(energies_by_batch, Eref_by_batch)
rsquare = r ** 2
print('R^2(Pearson correlation coefficient: {:.4}'.format(rsquare))
dct = {'Energy Reference(eV)': energies_by_batch, 'Energy Test(eV)': Eref_by_batch, 'Error(eV)': Error_by_mol_batch, 'Variance(eV)': var_by_batch,'sigma2':sigma2_by_batch}
df = pd.DataFrame(dct)
if save:
    df.to_csv('Results_t.csv', index=False)

if plot:
    fig, ax = plt.subplots()
    sns.regplot(x='Energy Reference(eV)', y='Energy Test(eV)', data=df,ax=ax)
    ax.errorbar(energies_by_batch, Eref_by_batch, yerr=var_by_batch, fmt='none',capsize=5, zorder=1,color='C0')
    ax.text(0.7, 0.1, (r'$[%.2f,%.2f,%.2f]$' % (rsquare,MAE_final, RMSE_final)), transform=ax.transAxes, fontsize=10,
             verticalalignment='top')
    plt.show()
    if save_plot:
        fig.savefig('evidential_test_set_scale.pdf', bbox_inches='tight')


