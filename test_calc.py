# Standard imports
import torch
import numpy as np
import argparse
#ASE importations
import ase
from ase.calculators.calculator import Calculator
from ase.io import read
#Neural network imports
from Neural_Net_evid import PhysNet

#Parser

parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

# Add arguments
parser.add_argument("--restart", type=str, default='No')
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
parser.add_argument("--summary_interval", default=5, type=int)
parser.add_argument("--validation_interval", default=5, type=int)
parser.add_argument("--show_progress", default=True, type=bool)
parser.add_argument("--save_interval", default=5, type=int)
parser.add_argument("--record_run_metadata", default=0, type=int)
parser.add_argument('--device', default='cpu', type=str)

# Read config file
config = 'input.inp'
args = parser.parse_args(["@" + config])

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
    activation_fn="shift_softplus",
    device=args.device)

checkpoint = torch.load('best_model.pt',map_location=torch.device('cpu'))
# print(checkpoint["state_dict"])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


def get_indices(Nref,device='cpu'):
    # Indices pointing to atom at each batch image
    idx = torch.arange(end=Nref.item(), dtype=torch.int32).to(device)
    # Indices for atom pairs ij - Atom i
    Nref_tot = torch.tensor(0, dtype=torch.int32).to(device)
    Ntmp = Nref.cpu()
    idx_i = idx.repeat(int(Ntmp.numpy()) - 1) + Nref_tot
    # Indices for atom pairs ij - Atom j
    idx_j = torch.roll(idx, -1, dims=0) + Nref_tot
    for Na in torch.arange(2, Nref.item()):
        Na_tmp = Na.cpu()
        idx_j = torch.concat(
            [idx_j, torch.roll(idx, int(-Na_tmp.numpy()), dims=0) + Nref_tot],
            dim=0)

    # Increment auxiliary parameter
    Nref_tot = Nref_tot + Nref.item()

    idx_i = torch.sort(idx_i)[0]
    # Complete indices arrays
    for Nref_a in Nref:

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

    # Combine indices for batch image and respective atoms
    idx = torch.stack([batch_seg, idx], dim=1)
    return idx.type(torch.int64), idx_i.type(torch.int64), idx_j.type(torch.int64)

# def get_indices(atoms, device='cpu'):
#     # Number of atoms
#     N = len(atoms)
#
#     # Indices pointing to atom at each batch image
#     idx = torch.arange(end=N, dtype=torch.int32).to(device)
#     # Indices for atom pairs ij - Atom i
#     idx_i = idx.repeat(int(N) - 1)
#     # Indices for atom pairs ij - Atom j
#     idx_j = torch.roll(idx, -1, dims=0)
#
#     if N >= 2:
#         for Na in torch.arange(2, N):
#             Na_tmp = Na.cpu()
#             idx_j = torch.concat(
#                 [idx_j, torch.roll(idx, int(-Na_tmp.numpy()), dims=0)],
#                 dim=0)
#
#     return idx_i.to(device), idx_j.to(device)


file = 'a_395.xyz'
#read input file
atoms = read(file)

Z = torch.tensor(atoms.get_atomic_numbers(),dtype=torch.int32,device=args.device)
R = torch.tensor(atoms.get_positions(),dtype=torch.float32,device=args.device)
N = torch.tensor(len(atoms),dtype=torch.int32,device=args.device)
idx,idx_i, idx_j = get_indices(N, device=args.device)
# print('idx_i',torch.sort(idx_i)[0])
# print('idx_i shape', idx_i.shape)
# print('idx_j',idx_j)
# print('idx_j',idx_j.shape)
out1 = model.energy(Z, R, idx_i, idx_j)
print(out1)

#'attach' calculator to atoms object
# atoms.set_calculator(calc)

#print potential energy (to scalar to display more digits)

# e,var,sigma2 = atoms.get_potential_energy()


# print("Potential energy: %.8f eV" % e)
# print("Variance: %.8f eV" % var)