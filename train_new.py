# Standard importations
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
# For Time measurement
from shutil import copyfile
from datetime import datetime
from time import time
# Neural Network importations
from NeuralNet import PhysNet
# from neural_network.activation_fn import *
from NeuralNet import gather_nd
from DataContainer import DataContainer

# Configure logging environment
logging.basicConfig(filename='train.log', level=logging.DEBUG)

# ------------------------------------------------------------------------------
# Command line arguments
# ------------------------------------------------------------------------------

# Initiate parser
parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

# Add arguments
parser.add_argument("--restart", type=str, default='No',
                    help="Restart training from a specific folder")
parser.add_argument("--checkpoint-file", type=str, default=None,
                    help="File to be loaded if model is restarted")
parser.add_argument("--num_features", default=128, type=int,
                    help="Dimensionality of feature vectors")
parser.add_argument("--num_basis", default=64, type=int,
                    help="Number of radial basis functions")
parser.add_argument("--num_blocks", default=5, type=int,
                    help="Number of interaction blocks")
parser.add_argument("--num_residual_atomic", default=2, type=int,
                    help="Number of residual layers for atomic refinements")
parser.add_argument("--num_residual_interaction", default=3, type=int,
                    help="Number of residual layers for the message phase")
parser.add_argument("--num_residual_output", default=1, type=int,
                    help="Number of residual layers for the output blocks")
parser.add_argument("--cutoff", default=10.0, type=float,
                    help="Cutoff distance for short range interactions")
parser.add_argument("--use_electrostatic", default=1, type=int,
                    help="Use electrostatics in energy prediction (0/1)")
parser.add_argument("--use_dispersion", default=1, type=int,
                    help="Use dispersion in energy prediction (0/1)")
parser.add_argument("--grimme_s6", default=None, type=float,
                    help="Grimme s6 dispersion coefficient")
parser.add_argument("--grimme_s8", default=None, type=float,
                    help="Grimme s8 dispersion coefficient")
parser.add_argument("--grimme_a1", default=None, type=float,
                    help="Grimme a1 dispersion coefficient")
parser.add_argument("--grimme_a2", default=None, type=float,
                    help="Grimme a2 dispersion coefficient")
parser.add_argument("--dataset", type=str,
                    help="File path to dataset")
# This number is configured for the size of the dataset for Sn2 reeactions
parser.add_argument("--num_train", default=362167, type=int,
                    help="Number of training samples")
# This number is configured for the size of the dataset for Sn2 reeactions
parser.add_argument("--num_valid", default=45000, type=int,
                    help="Number of validation samples")
parser.add_argument("--batch_size", default=100, type=int,
                    help="Batch size used per training step")
parser.add_argument("--valid_batch_size", default=20, type=int,
                    help="Batch size used for going through validation_set")
parser.add_argument("--seed", default=np.random.randint(1000000), type=int,
                    help="Seed for splitting dataset into " + \
                         "training/validation/test")
parser.add_argument("--max_steps", default=10000, type=int,
                    help="Maximum number of training steps")
parser.add_argument("--learning_rate", default=0.001, type=float,
                    help="Learning rate used by the optimizer")
parser.add_argument("--decay_steps", default=1000, type=int,
                    help="Decay the learning rate every N steps by decay_rate")
parser.add_argument("--decay_rate", default=0.1, type=float,
                    help="Factor with which the learning rate gets " + \
                         "multiplied by every decay_steps steps")
# TODO: Check how to add this
parser.add_argument("--max_norm", default=1000.0, type=float,
                    help="Max norm for gradient clipping")
# TODO: Check how to add this
parser.add_argument("--ema_decay", default=0.999, type=float,
                    help="Exponential moving average decay used by the " + \
                         "trainer")
parser.add_argument("--rate", default=0.0, type=float,
                    help="Rate probability for dropout regularization of " + \
                         "rbf layer")
parser.add_argument("--l2lambda", default=0.0, type=float,
                    help="Lambda multiplier for l2 loss (regularization)")
parser.add_argument("--nhlambda", default=0.1, type=float,
                    help="Lambda multiplier for non-hierarchicality " + \
                         "loss (regularization)")
parser.add_argument('--force_weight', default=52.91772105638412, type=float,
                    help="This defines the force contribution to the loss " + \
                         "function relative to the energy contribution (to" + \
                         " take into account the different numerical range)")
parser.add_argument('--charge_weight', default=14.399645351950548, type=float,
                    help="This defines the charge contribution to the " + \
                         "loss function relative to the energy " + \
                         "contribution (to take into account the " + \
                         "different  numerical range)")
parser.add_argument('--dipole_weight', default=27.211386024367243, type=float,
                    help="This defines the dipole contribution to the " + \
                         "loss function relative to the energy " + \
                         "contribution (to take into account the " + \
                         "different numerical range)")
# TODO: Check if this is still needed
parser.add_argument('--summary_interval', default=5, type=int,
                    help="Write a summary every N steps")
parser.add_argument('--validation_interval', default=5, type=int,
                    help="Check performance on validation set every N steps")
parser.add_argument('--show_progress', default=True, type=bool,
                    help="Show progress of the epoch")
parser.add_argument('--save_interval', default=5, type=int,
                    help="Save progress every N steps")
parser.add_argument('--record_run_metadata', default=0, type=int,
                    help="Records metadata like memory consumption etc.")

# ------------------------------------------------------------------------------
# Read Parameters and define output files
# ------------------------------------------------------------------------------

# Generate an (almost) unique id for the training session
def id_generator(size=8,
                 chars=(string.ascii_uppercase
                        + string.ascii_lowercase
                        + string.digits)):
    return ''.join(random.SystemRandom().choice(chars) for _ in range(size))


# Read config file if no arguments are given
config_file = 'config.txt'
if len(sys.argv) == 1:
    if os.path.isfile(config_file):
        args = parser.parse_args(["@" + config_file])
    else:
        args = parser.parse_args(["--help"])
else:
    args = parser.parse_args()

# Create output directory for training session and
# load config file arguments if restart
if args.restart == 'No':
    directory = (
            datetime.utcnow().strftime("%Y%m%d%H%M%S")
            + "_" + id_generator() + "_F" + str(args.num_features)
            + "K" + str(args.num_basis) + "b" + str(args.num_blocks)
            + "a" + str(args.num_residual_atomic)
            + "i" + str(args.num_residual_interaction)
            + "o" + str(args.num_residual_output) + "cut" + str(args.cutoff)
            + "e" + str(args.use_electrostatic) + "d" + str(args.use_dispersion)
            + "l2" + str(args.l2lambda) + "nh" + str(args.nhlambda)
            + "rate" + str(args.rate))
    checkpoint_file = args.checkpoint_file
else:
    directory = args.restart
    args = parser.parse_args(["@" + os.path.join(args.restart, config_file)])
    checkpoint_file = os.path.join(args.restart, args.checkpoint_file)

# Create sub directories
logging.info("Creating directories...")

if not os.path.exists(directory):
    os.makedirs(directory)

best_dir = os.path.join(directory, 'best')
if not os.path.exists(best_dir):
    os.makedirs(best_dir)

log_dir = os.path.join(directory, 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
ckpt_dir = os.path.join(directory, 'ckpt')
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

# Define output files
best_loss_file = os.path.join(best_dir, 'best_loss.npz')
best_checkpoint = os.path.join(best_dir, 'best_model.pt')

# Write config file of current training session
logging.info("Writing args to file...")

with open(os.path.join(directory, config_file), 'w') as f:
    for arg in vars(args):
        f.write('--' + arg + '=' + str(getattr(args, arg)) + "\n")

# ------------------------------------------------------------------------------
# Define utility functions
# ------------------------------------------------------------------------------


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

    # Combine indices for batch image and respective atoms
    idx = torch.stack([batch_seg, idx], dim=1)
    return idx.type(torch.int64), idx_i.type(torch.int64), idx_j.type(torch.int64), batch_seg.type(torch.int64)

def reset_averages():
    ''' Reset counter and average values '''

    null_float = torch.tensor(0.0, dtype=torch.float32)

    return null_float, null_float, null_float, null_float, null_float, \
           null_float, null_float, null_float, null_float, null_float
# ------------------------------------------------------------------------------
# Load data and initiate PhysNet model
# ------------------------------------------------------------------------------

# Load dataset
logging.info("Loading dataset...")

data = DataContainer(
    args.dataset, args.num_train, args.num_valid,
    args.batch_size, args.valid_batch_size, seed=args.seed)

# Initiate PhysNet model
logging.info("Creating PhysNet model...")

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
    Eshift=data.EperA_m_n,
    Escale=data.EperA_s_n,
    activation_fn="shift_softplus",
    device='cuda')


# ------------------------------------------------------------------------------
# Start the training step
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Train PhysNet model
# ------------------------------------------------------------------------------

logging.info("starting training...")

# Define Optimizer
optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate,
                             weight_decay=args.l2lambda, amsgrad=True)

lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay_rate)

# Initiate epoch and step counter
epoch = torch.tensor(1, requires_grad=False, dtype=torch.int64)
step = torch.tensor(1, requires_grad=False, dtype=torch.int64)

# Create validation batches
valid_batches = data.get_valid_batches()

# Initialize counter for estimated time per epoch
time_train_estimation = np.nan
time_train = 0.0

lf = nn.L1Loss(reduction="mean")
device = "cuda"
for epoch in range(args.max_steps):
    # Reset error averages
    num_t, loss_avg_t, emse_avg_t, emae_avg_t, fmse_avg_t, fmae_avg_t, \
    qmse_avg_t, qmae_avg_t, dmse_avg_t, dmae_avg_t = reset_averages()

    model.train()
    # Create train batches
    train_batches, N_train_batches = data.get_train_batches()

    # Start train timer
    train_start = time()

    # Iterate over batches
    for ib, batch in enumerate(train_batches):
        optimizer.zero_grad()
        model.train()
        # Decompose data
        batch = [i.to(device) for i in batch]
        N_t, Z_t, R_t, Eref_t, Earef_t, Fref_t, Qref_t, Qaref_t, Dref_t = batch

        # Get indices
        indices = [i.to(device) for i in get_indices(N_t, device="cuda")]
        idx_t, idx_i_t, idx_j_t, batch_seg_t = indices
        # Gather data
        Z_t = gather_nd(Z_t, idx_t)
        R_t = gather_nd(R_t, idx_t)

        if torch.count_nonzero(Earef_t) != 0:
            Earef_t = gather_nd(Earef_t, idx_t)
        if torch.count_nonzero(Fref_t) != 0:
            Fref_t = gather_nd(Fref_t, idx_t)
        if torch.count_nonzero(Qaref_t) != 0:
            Qaref_t = gather_nd(Qaref_t, idx_t)

        # Evaluate model
        energy_t, forces_t, Ea_t, Qa_t, nhloss_t = \
            model.energy_and_forces_and_atomic_properties(
                Z_t, R_t, idx_i_t, idx_j_t, Qref_t, batch_seg_t)

        u_bst = len(torch.unique(batch_seg_t))
        Qtot_t = Qa_t.new_zeros(u_bst).index_add(0, batch_seg_t.type(torch.int64), Qa_t)
        QR_t = torch.stack([Qa_t * R_t[:, 0], Qa_t * R_t[:, 1], Qa_t * R_t[:, 2]], 1)

        D_t = torch.zeros(u_bst, 3,device=device).index_add(0, batch_seg_t.type(torch.int64), QR_t)

        loss_t = lf(Eref_t,energy_t) + lf(Fref_t, forces_t) + lf(Qref_t, Qtot_t) + lf(Dref_t, D_t)

        loss_t.backward(retain_graph=True)
        optimizer.step()
        print(loss_t)