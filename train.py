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
from torch_ema import ExponentialMovingAverage
# For Time measurement
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
parser.add_argument("--max_norm", default=1000.0, type=float,
                    help="Max norm for gradient clipping")
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
parser.add_argument('--device',default='cuda',type=str,
                    help='Selects the device that will be used for training')

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


# Write config file of current training session
logging.info("Writing args to file...")

with open(os.path.join(directory, config_file), 'w') as f:
    for arg in vars(args):
        f.write('--' + arg + '=' + str(getattr(args, arg)) + "\n")

logging.info("device: {}".format(args.device))

# ------------------------------------------------------------------------------
# Define utility functions
# ------------------------------------------------------------------------------
def save_checkpoint(model, epoch, optimizer, name_of_ckpt=None,best=False):
    state = {'model_state_dict': model.state_dict(),
             'epoch': epoch, 'optimizer_state_dict': optimizer.state_dict()}
    if best:
        path = os.path.join(best_dir, 'best_model.pt')
    else:
        name = 'model' + str(name_of_ckpt) + '.pt'
        path = os.path.join(ckpt_dir, name)

    torch.save(state, path)


def load_checkpoint(path):
    if path is not None:
        checkpoint = torch.load(path)
        return checkpoint
    else:
        return None


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='#', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(
        100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print("\r{0} |{1}| {2}% {3}".format(
        prefix, bar, percent, suffix), end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def reset_averages(device='cpu'):
    ''' Reset counter and average values '''

    null_float = torch.tensor(0.0, dtype=torch.float32,device=device)

    return null_float, null_float, null_float, null_float, null_float, \
           null_float, null_float, null_float, null_float, null_float


def calculate_errors(val1, val2):
    ''' Calculate error values and loss function '''
    lf = nn.L1Loss(reduction="mean")
    loss = lf(val1, val2)
    delta2 = loss ** 2
    mae = loss
    # Mean squared error
    mse = torch.sum(delta2)

    return loss, mse, mae

#Should this go also to cuda?
def calculate_null(val1, val2,device='cpu'):
    ''' Return zero for error and loss values '''

    null = torch.zeros(1, dtype=torch.float32,device=device)

    return null, null, null


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

# Initiate summary writer
summary_writer = SummaryWriter(log_dir)
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
    device=args.device,
    writer = summary_writer)

# ------------------------------------------------------------------------------
# Prepare loss evaluation and model trainer
# ------------------------------------------------------------------------------

logging.info("prepare training...")

# Set evaluation function loss and error values if reference data are
# available (return zero otherwise) for ...
# Total energy
if data.include_E:
    e_eval = calculate_errors
else:
    e_eval = calculate_null
# Atomic energy
if data.include_Ea:
    ea_eval = calculate_errors
else:
    ea_eval = calculate_null
# Forces
if data.include_F:
    f_eval = calculate_errors
else:
    f_eval = calculate_null
# Total charge
if data.include_Q:
    q_eval = calculate_errors
else:
    q_eval = calculate_null
# Atomic charges
if data.include_Qa:
    qa_eval = calculate_errors
else:
    qa_eval = calculate_null
# Dipole moment
if data.include_D:
    d_eval = calculate_errors
else:
    d_eval = calculate_null

# Load best recorded loss if available
if os.path.isfile(best_loss_file):
    loss_file = np.load(best_loss_file)
    best_loss = loss_file["loss"].item()
    best_emae = loss_file["emae"].item()
    best_ermse = loss_file["ermse"].item()
    best_fmae = loss_file["fmae"].item()
    best_frmse = loss_file["frmse"].item()
    best_qmae = loss_file["qmae"].item()
    best_qrmse = loss_file["qrmse"].item()
    best_dmae = loss_file["dmae"].item()
    best_drmse = loss_file["drmse"].item()
    best_epoch = loss_file["epoch"].item()
else:
    best_loss = np.Inf
    best_emae = np.Inf
    best_ermse = np.Inf
    best_fmae = np.Inf
    best_frmse = np.Inf
    best_qmae = np.Inf
    best_qrmse = np.Inf
    best_dmae = np.Inf
    best_drmse = np.Inf
    best_epoch = 0.
    np.savez(
        best_loss_file, loss=best_loss, emae=best_emae, ermse=best_ermse,
        fmae=best_fmae, frmse=best_frmse, qmae=best_qmae, qrmse=best_qrmse,
        dmae=best_dmae, drmse=best_drmse, epoch=best_epoch)


# ------------------------------------------------------------------------------
# Define training step
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


def train_step(batch, num_t, loss_avg_t, emse_avg_t, emae_avg_t, fmse_avg_t, fmae_avg_t,
               qmse_avg_t, qmae_avg_t, dmse_avg_t, dmae_avg_t,device='cpu'):
    # Initialize training mode
    model.train()
    # Decompose data
    batch = [i.to(device) for i in batch]
    N_t, Z_t, R_t, Eref_t, Earef_t, Fref_t, Qref_t, Qaref_t, Dref_t = batch

    # Get indices
    indices = [i.to(device) for i in get_indices(N_t,device=device)]
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
    Qtot_t = Qa_t.new_zeros(u_bst).index_add(0, batch_seg_t.type(torch.int64), Qa_t).to(device)
    QR_t = torch.stack([Qa_t * R_t[:, 0], Qa_t * R_t[:, 1], Qa_t * R_t[:, 2]], 1)

    D_t = torch.zeros(u_bst, 3,device=device).index_add(0, batch_seg_t.type(torch.int64), QR_t)

    # Evaluate error and losses for ...
    eloss_t, emse_t, emae_t = e_eval(Eref_t, energy_t)
    # # Atomic energy
    ealoss_t, eamse_t, eamae_t = ea_eval(Earef_t, Ea_t,device=device)
    # # Forces
    floss_t, fmse_t, fmae_t = f_eval(Fref_t, forces_t)
    # # Total charge
    qloss_t, qmse_t, qmae_t = q_eval(Qref_t, Qtot_t)
    # # Atomic charges (Is this necessary?)
    qaloss_t, qamse_t, qamae_t = qa_eval(Qaref_t, Qa_t,device=device)
    # # Dipole moment
    dloss_t, dmse_t, dmae_t = d_eval(Dref_t, D_t)

    loss_t = (eloss_t + ealoss_t
              + args.force_weight * floss_t
              + args.charge_weight * qloss_t
              + args.dipole_weight * dloss_t
              + args.nhlambda * nhloss_t)

    loss_t.backward(retain_graph=True)
    #Gradient clip
    nn.utils.clip_grad_norm_(model.parameters(),args.max_norm)

    f = num_t / (num_t + N_t.dim())

    loss_avg_t = f * loss_avg_t + (1.0 - f) * float(loss_t)
    emse_avg_t = f * emse_avg_t + (1.0 - f) * float(emse_t)
    emae_avg_t = f * emae_avg_t + (1.0 - f) * float(emae_t)
    fmse_avg_t = f * fmse_avg_t + (1.0 - f) * float(fmse_t)
    fmae_avg_t = f * fmae_avg_t + (1.0 - f) * float(fmae_t)
    qmse_avg_t = f * qmse_avg_t + (1.0 - f) * float(qmse_t)
    qmae_avg_t = f * qmae_avg_t + (1.0 - f) * float(qmae_t)
    dmse_avg_t = f * dmse_avg_t + (1.0 - f) * float(dmse_t)
    dmae_avg_t = f * dmae_avg_t + (1.0 - f) * float(dmae_t)

    num_t = num_t + N_t.dim()

    return num_t, loss_avg_t, emse_avg_t, emae_avg_t, fmse_avg_t, fmae_avg_t, \
           qmse_avg_t, qmae_avg_t, dmse_avg_t, dmae_avg_t


def valid_step(batch, num_v, loss_avg_v, emse_avg_v, emae_avg_v, fmse_avg_v, fmae_avg_v,
               qmse_avg_v, qmae_avg_v, dmse_avg_v, dmae_avg_v,device='cpu'):
    # Intialize validation mode
    model.eval()

    # Decompose data
    batch = [i.to(device) for i in batch]
    N_v, Z_v, R_v, Eref_v, Earef_v, Fref_v, Qref_v, Qaref_v, Dref_v = batch

    # Get indices
    idx_v, idx_i_v, idx_j_v, batch_seg_v = get_indices(N_v,device=device)

    # Gather data
    Z_v = gather_nd(Z_v, idx_v)
    R_v = gather_nd(R_v, idx_v)

    if torch.count_nonzero(Earef_v) != 0:
        Earef_v = gather_nd(Earef_v, idx_v)
    if torch.count_nonzero(Fref_v) != 0:
        Fref_v = gather_nd(Fref_v, idx_v)
    if torch.count_nonzero(Qaref_v) != 0:
        Qaref_v = gather_nd(Qaref_v, idx_v)

    # Calculate quantities
    # with torch.no_grad():

    energy_v, forces_v, Ea_v, Qa_v, nhloss_v = \
        model.energy_and_forces_and_atomic_properties(
            Z_v, R_v, idx_i_v, idx_j_v, Qref_v, batch_seg_v)

    # Get total charge
    u_bsv = len(torch.unique(batch_seg_v))
    Qtot_v = Qa_v.new_zeros(u_bsv).index_add(0, batch_seg_v.type(torch.int64), Qa_v).to(device)

    # Get dipole moment vector
    QR_v = torch.stack([Qa_v * R_v[:, 0], Qa_v * R_v[:, 1], Qa_v * R_v[:, 2]], 1)
    D_v = torch.zeros(u_bsv, 3,device=device).index_add(0, batch_seg_v.type(torch.int64), QR_v)

    # Evaluate error and losses for ...
    # Total energy
    eloss_v, emse_v, emae_v = e_eval(Eref_v, energy_v)
    # Atomic energy
    ealoss_v, eamse_v, eamae_v = ea_eval(Earef_v, Ea_v,device=device)
    # Forces
    floss_v, fmse_v, fmae_v = f_eval(Fref_v, forces_v)
    # Total charge
    qloss_v, qmse_v, qmae_v = q_eval(Qref_v, Qtot_v)
    # Atomic charges
    qaloss_v, qamse_v, qamae_v = qa_eval(Qaref_v, Qa_v,device=device)
    # Dipole moment
    dloss_v, dmse_v, dmae_v = d_eval(Dref_v, D_v)

    # Evaluate total loss
    loss_v = (eloss_v + ealoss_v
              + args.force_weight * floss_v
              + args.charge_weight * qloss_v
              + args.dipole_weight * dloss_v
              + args.nhlambda * nhloss_v)
    loss_v.backward(retain_graph=True)

    # Update averages
    f = num_v / (num_v + N_v.dim())

    loss_avg_v = f * loss_avg_v + (1.0 - f) * float(loss_v)
    emse_avg_v = f * emse_avg_v + (1.0 - f) * float(emse_v)
    emae_avg_v = f * emae_avg_v + (1.0 - f) * float(emae_v)
    fmse_avg_v = f * fmse_avg_v + (1.0 - f) * float(fmse_v)
    fmae_avg_v = f * fmae_avg_v + (1.0 - f) * float(fmae_v)
    qmse_avg_v = f * qmse_avg_v + (1.0 - f) * float(qmse_v)
    qmae_avg_v = f * qmae_avg_v + (1.0 - f) * float(qmae_v)
    dmse_avg_v = f * dmse_avg_v + (1.0 - f) * float(dmse_v)
    dmae_avg_v = f * dmae_avg_v + (1.0 - f) * float(dmae_v)
    num_v = num_v + N_v.dim()

    return num_v, loss_avg_v, emse_avg_v, emae_avg_v, fmse_avg_v, fmae_avg_v, \
           qmse_avg_v, qmae_avg_v, dmse_avg_v, dmae_avg_v


# ------------------------------------------------------------------------------
# Train PhysNet model
# ------------------------------------------------------------------------------

logging.info("starting training...")

# Define Optimizer
optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate,
                             weight_decay=args.l2lambda, amsgrad=True)

lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay_rate)

# Define Exponential Moving Average

ema = ExponentialMovingAverage(model.parameters(),decay=args.ema_decay)
# Initiate epoch and step counter
epoch = torch.tensor(1, requires_grad=False, dtype=torch.int64)
step = torch.tensor(1, requires_grad=False, dtype=torch.int64)

# Initiate checkpoints and load latest checkpoint

latest_ckpt = load_checkpoint(checkpoint_file)
if latest_ckpt is not None:
    model.load_state_dict(latest_ckpt['model_state_dict'])
    optimizer.load_state_dict(latest_ckpt['optimizer_state_dict'])
    epoch = latest_ckpt['epoch']

# Create validation batches
valid_batches = data.get_valid_batches()

# Initialize counter for estimated time per epoch
time_train_estimation = np.nan
time_train = 0.0

# Training loop
# Terminate training when maximum number of iterations is reached
while epoch <= args.max_steps:

    # Reset error averages
    num_t, loss_avg_t, emse_avg_t, emae_avg_t, fmse_avg_t, fmae_avg_t, \
    qmse_avg_t, qmae_avg_t, dmse_avg_t, dmae_avg_t = reset_averages(device=args.device)


    # Create train batches
    train_batches, N_train_batches = data.get_train_batches()

    # Start train timer
    train_start = time()

    # Iterate over batches
    for ib, batch in enumerate(train_batches):
        optimizer.zero_grad()
        # Start batch timer
        batch_start = time()

        # Show progress bar
        if args.show_progress:
            printProgressBar(
                ib, N_train_batches, prefix="Epoch {0: 5d}".format(
                    epoch.numpy()),
                suffix=("Complete - Remaining Epoch Time: "
                        + "{0: 4.1f} s     ".format(time_train_estimation)),
                length=42)

        # Training step
        num_t, loss_avg_t, emse_avg_t, emae_avg_t, fmse_avg_t, fmae_avg_t, \
        qmse_avg_t, qmae_avg_t, dmse_avg_t, dmae_avg_t = train_step(
            batch, num_t, loss_avg_t, emse_avg_t, emae_avg_t,
            fmse_avg_t, fmae_avg_t, qmse_avg_t, qmae_avg_t,
            dmse_avg_t, dmae_avg_t,device=args.device)

        optimizer.step()
        ema.update()
        # Stop batch timer
        batch_end = time()

        # Actualize time estimation
        if args.show_progress:
            if ib == 0:
                time_train_estimation = (
                        (batch_end - batch_start) * (N_train_batches - 1))
            else:
                time_train_estimation = (
                        0.5 * (time_train_estimation - (batch_end - batch_start))
                        + 0.5 * (batch_end - batch_start) * (N_train_batches - ib - 1))

        # Increment step number
        step = step + 1

    # Stop train timer
    train_end = time()
    time_train = train_end - train_start

    # Show final progress bar and time
    if args.show_progress:
        loss_avg_t_temp = loss_avg_t.detach().cpu()
        lat = float(loss_avg_t_temp.numpy())
        printProgressBar(
            N_train_batches, N_train_batches, prefix="Epoch {0: 5d}".format(
                epoch.numpy()),
            suffix=("Done - Epoch Time: "
                    + "{0: 4.1f} s, Average Loss: {1: 4.4f}   ".format(
                        time_train, lat)))  # length=42))

    # Save progress
    if (epoch % args.save_interval == 0):
        number_of_ckpt = int(epoch / args.save_interval)
        save_checkpoint(model=model, epoch=epoch, optimizer=optimizer, name_of_ckpt=number_of_ckpt)


    # Check performance on the validation set

    if (epoch % args.validation_interval) == 0:
        # Update training results
        results_t = {}
        loss_avg_t_temp = loss_avg_t.detach().cpu()
        results_t["loss_train"] = loss_avg_t_temp.numpy()
        results_t["time_train"] = time_train
        if data.include_E:
            emae_avg_t_temp = emae_avg_t.detach().cpu()
            results_t["energy_mae_train"] = emae_avg_t_temp.numpy()
            results_t["energy_rmse_train"] = np.sqrt(emae_avg_t_temp.numpy())
        if data.include_F:
            fmae_avg_t_temp = fmae_avg_t.detach().cpu()
            results_t["forces_mae_train"] = fmae_avg_t_temp.numpy()
            results_t["forces_rmse_train"] = np.sqrt(fmae_avg_t_temp.numpy())
        if data.include_Q:
            qmae_avg_t_temp = qmae_avg_t.detach().cpu()
            results_t["charge_mae_train"] = qmae_avg_t_temp.numpy()
            results_t["charge_rmse_train"] = np.sqrt(qmae_avg_t_temp.numpy())
        if data.include_D:
            dmae_avg_t_temp = dmae_avg_t.detach().cpu()
            results_t["dipole_mae_train"] = dmae_avg_t_temp.numpy()
            results_t["dipole_rmse_train"] = np.sqrt(dmae_avg_t_temp.numpy())

        # Write Results to tensorboard
        for key, value in results_t.items():
            summary_writer.add_scalar(key, value, global_step=epoch)

        # # Backup variables and assign EMA variables
        # backup_vars = [tf.identity(var) for var in model.trainable_variables]
        # for var in model.trainable_variables:
        #     var.assign(ema.average(var))

        # Reset error averages
        num_v, loss_avg_v, emse_avg_v, emae_avg_v, fmse_avg_v, fmae_avg_v, \
        qmse_avg_v, qmae_avg_v, dmse_avg_v, dmae_avg_v = reset_averages()

        # Start valid timer
        valid_start = time()

        for ib, batch in enumerate(valid_batches):
            model.eval()
            num_v, loss_avg_v, emse_avg_v, emae_avg_v, \
            fmse_avg_v, fmae_avg_v, qmse_avg_v, qmae_avg_v, \
            dmse_avg_v, dmae_avg_v = valid_step(
                batch, num_v, loss_avg_v, emse_avg_v, emae_avg_v,
                fmse_avg_v, fmae_avg_v, qmse_avg_v, qmae_avg_v,
                dmse_avg_v, dmae_avg_v,device=args.device)

        # Stop valid timer
        valid_end = time()
        time_valid = valid_end - valid_end

        # Update validation results
        results_v = {}
        loss_avg_v_temp = loss_avg_v.detach().cpu()
        results_v["loss_valid"] = loss_avg_v_temp.numpy()
        results_t["time_valid"] = time_valid
        if data.include_E:
            emae_avg_v_temp = emae_avg_v.detach().cpu()
            results_v["energy_mae_valid"] = emae_avg_v_temp.numpy()
            results_v["energy_rmse_valid"] = np.sqrt(emae_avg_v_temp.numpy())
        if data.include_F:
            fmae_avg_v_temp = fmae_avg_v.detach().cpu()
            results_v["forces_mae_valid"] = fmae_avg_v_temp.numpy()
            results_v["forces_rmse_valid"] = np.sqrt(fmae_avg_v_temp.numpy())
        if data.include_Q:
            qmae_avg_v_temp = qmae_avg_v.detach().cpu()
            results_v["charge_mae_valid"] = qmae_avg_v_temp.numpy()
            results_v["charge_rmse_valid"] = np.sqrt(qmae_avg_v_temp.numpy())
        if data.include_D:
            dmae_avg_v_temp = dmae_avg_v.detach().cpu()
            results_v["dipole_mae_valid"] = dmae_avg_v_temp.numpy()
            results_v["dipole_rmse_valid"] = np.sqrt(dmae_avg_v_temp.numpy())

        for key, value in results_v.items():
            summary_writer.add_scalar(key, value, global_step=epoch)

        if results_v["loss_valid"] < best_loss:

            # Assign results of best validation
            best_loss = results_v["loss_valid"]
            if data.include_E:
                best_emae = results_v["energy_mae_valid"]
                best_ermse = results_v["energy_rmse_valid"]
            else:
                best_emae = np.Inf
                best_ermse = np.Inf
            if data.include_F:
                best_fmae = results_v["forces_mae_valid"]
                best_frmse = results_v["forces_rmse_valid"]
            else:
                best_fmae = np.Inf
                best_frmse = np.Inf
            if data.include_Q:
                best_qmae = results_v["charge_mae_valid"]
                best_qrmse = results_v["charge_rmse_valid"]
            else:
                best_qmae = np.Inf
                best_qrmse = np.Inf
            if data.include_D:
                best_dmae = results_v["dipole_mae_valid"]
                best_drmse = results_v["dipole_rmse_valid"]
            else:
                best_dmae = np.Inf
                best_drmse = np.Inf
            best_epoch = epoch.numpy()

            # Save best results
            np.savez(
                best_loss_file, loss=best_loss,
                emae=best_emae, ermse=best_ermse,
                fmae=best_fmae, frmse=best_frmse,
                qmae=best_qmae, qrmse=best_qrmse,
                dmae=best_dmae, drmse=best_drmse,
                epoch=best_epoch)

            # Save best model variables
            save_checkpoint(model=model, epoch=epoch, optimizer=optimizer,best=True)

        # Update best results
        results_b = {}
        results_b["loss_best"] = best_loss
        if data.include_E:
            results_b["energy_mae_best"] = best_emae
            results_b["energy_rmse_best"] = best_ermse
        if data.include_F:
            results_b["forces_mae_best"] = best_fmae
            results_b["forces_rmse_best"] = best_frmse
        if data.include_Q:
            results_b["charge_mae_best"] = best_qmae
            results_b["charge_rmse_best"] = best_qrmse
        if data.include_D:
            results_b["dipole_mae_best"] = best_dmae
            results_b["dipole_rmse_best"] = best_drmse

        # Write the results to tensorboard
        for key, value in results_b.items():
            summary_writer.add_scalar(key, value, global_step=epoch)


        # for var, bck in zip(model.trainable_variables, backup_vars):
        #     var.assign(bck)

    # Generate summaries
    if ((epoch % args.summary_interval == 0)
            and (epoch >= args.validation_interval)):

        if data.include_E:
            print(
                "Summary Epoch: " + \
                str(epoch.numpy()) + '/' + str(args.max_steps),
                "\n    Loss   train/valid: {0: 1.3e}/{1: 1.3e}, ".format(
                    results_t["loss_train"],
                    results_v["loss_valid"]),
                " Best valid loss:   {0: 1.3e}, ".format(
                    results_b["loss_best"]),
                "\n    MAE(E) train/valid: {0: 1.3e}/{1: 1.3e}, ".format(
                    results_t["energy_mae_train"],
                    results_v["energy_mae_valid"]),
                " Best valid MAE(E): {0: 1.3e}, ".format(
                    results_b["energy_mae_best"]))

    # Increment epoch number
    epoch += 1
