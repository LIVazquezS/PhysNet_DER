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
parser.add_argument("--summary_interval", default=5, type=int)
parser.add_argument("--validation_interval", default=5, type=int)
parser.add_argument("--show_progress", default=True, type=bool)
parser.add_argument("--save_interval", default=5, type=int)
parser.add_argument("--record_run_metadata", default=0, type=int)
parser.add_argument('--device', default='cuda', type=str)

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

checkpoints = ["./best_model.pt"]
def load_checkpoint(checkpoints):
    if len(checkpoints) == 1:
        if checkpoints[0] is not None:
            checkpoint = torch.load(checkpoints[0])
            return checkpoint
    else:
        for i in checkpoints:
            if i is not None:
                checkpoint = torch.load(i)
                return checkpoint
    return None


# Load neural network parameter
latest_ckpt = load_checkpoint(checkpoints)
# if len(latest_ckpt) == 1:
model.load_state_dict(latest_ckpt['model_state_dict'])
# else:
#     models = []
#     for i in latest_ckpt:
#         p = model.load_state_dict(i['model_state_dict'])
#         models.append(p)


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
    return energy_test_list, Eref_test_list, mae_energy, mse_energy, rmse_energy, var

# load dataset
data = DataContainer(
    args.dataset, args.num_train, args.num_valid,
    args.batch_size, args.valid_batch_size, seed=args.seed)

test_batches = data.get_test_batches()

for ib, batch in enumerate(test_batches):
    energy_test_list, Eref_test_list, mae_energy, mse_energy, rmse_energy, var = evid_test_step(batch,args.device)
    print('MAE(kcal/mol): {:.4}'.format(mae_energy*23))
    print('RMSE(kcal/mol): {:.4}'.format(rmse_energy*23))
    E_by_mol = np.abs(energy_test_list - Eref_test_list)
    dct = {'Energy Reference(eV)': Eref_test_list, 'Energy Test(eV)': energy_test_list, 'Error(eV)': E_by_mol, 'Variance(eV)': var}
    df = pd.DataFrame(dct)
    fig, ax = plt.subplots()
    sns.regplot(x='Energy Reference(eV)', y='Energy Test(eV)', data=df,ax=ax)
    ax.errorbar(energy_test_list, Eref_test_list, yerr=var, fmt='none',capsize=5, zorder=1,color='C0')
    plt.show()
    #Optional
    df.to_csv('Results_on_test_set.csv', index=False)




#
#
# # calculate errors
# _, emse, emae = calculate_errors(Eref, energy)
# _, fmse, fmae = calculate_errors(Fref, forces)
# _, dmse, dmae = calculate_errors(Dref, D)
# _, qmse, qmae = calculate_errors(Qref, Qtot)
#
#
#
#
# def generate_dataset_with_atomic_quantities(sess, data, checkpoints):
#     print("\n Data count: " + str(len(data)))
#     if data.N is not None:
#         Ndata = np.zeros([len(data)], dtype=int)
#     if data.E is not None:
#         Edata = np.zeros([len(data)], dtype=float)
#         Eadata = np.zeros([len(data), data.N_max], dtype=float)
#     if data.Q is not None:
#         Qdata = np.zeros([len(data)], dtype=float)
#     if data.D is not None:
#         Ddata = np.zeros([len(data), 3], dtype=float)
#         Qadata = np.zeros([len(data), data.N_max], dtype=float)
#     if data.Z is not None:
#         Zdata = np.zeros([len(data), data.N_max], dtype=int)
#     if data.R is not None:
#         Rdata = np.zeros([len(data), data.N_max, 3], dtype=float)
#     if data.F is not None:
#         Fdata = np.zeros([len(data), data.N_max, 3], dtype=float)
#
#     for i in range(len(data)):
#         print(i)
#         datapoint = data[i]
#         Natom = len(datapoint["R"])
#         # store the current datapoint in the numpy array
#         if data.N is not None:
#             Ndata[i] = Natom
#         if data.E is not None:
#             Edata[i] = np.asarray(datapoint["E"])
#         if data.Q is not None:
#             Qdata[i] = np.asarray(datapoint["Q"])
#         if data.D is not None:
#             Ddata[i, :] = np.asarray(datapoint["D"])
#         if data.Z is not None:
#             Zdata[i, :Natom] = np.asarray(datapoint["Z"])
#         if data.R is not None:
#             Rdata[i, :Natom, :] = np.asarray(datapoint["R"])
#         if data.F is not None:
#             Fdata[i, :Natom, :] = np.asarray(datapoint["F"])
#
#         feed_dict = fill_feed_dict(datapoint)
#
#         # calculate average atomic energy prediction of ensemble
#         Ea_avg = np.zeros([data.N_max], dtype=float)
#         Ea_var = np.zeros([data.N_max], dtype=float)
#         num = 1
#         for checkpoint in checkpoints:
#             nn.restore(sess, checkpoint)
#             Etmp, Eatmp, Qatmp = sess.run([energy, Ea, Qa], feed_dict=feed_dict)
#             for a in range(len(Eatmp)):
#                 deltaEa = (Eatmp[a] - Ea_avg[a])
#                 Ea_avg[a] += deltaEa / num
#                 Ea_var[a] += deltaEa * (Eatmp[a] - Ea_avg[a])
#             num += 1
#         Ea_var /= num - 1
#         # calculate error and correct atomic energies accordingly
#         # print(datapoint["E"][0], np.sum(Ea_avg), Ea_avg[0:len(Eatmp)]) #before correction
#         error = np.sum(Ea_avg) - datapoint["E"][0]
#         weight = Ea_var / np.sum(Ea_var)  # weight according to variance
#         Ea_avg -= weight * error
#         Eadata[i, :Natom] = Ea_avg[:Natom]
#
#     np.savez_compressed("qm9_atomic.npz", N=Ndata, E=Edata, Ea=Eadata, Z=Zdata, R=Rdata)
#
#
# # helper function to print errors
# def print_errors_old(sess, get_data, data_count, data_name, checkpoints):
#     print("\n" + data_name + " (" + str(data_count) + "):")
#     emse_avg = 0.0
#     emae_avg = 0.0
#     fmse_avg = 0.0
#     fmae_avg = 0.0
#     qmse_avg = 0.0
#     qmae_avg = 0.0
#     dmse_avg = 0.0
#     dmae_avg = 0.0
#     with open(data_name + ".csv", "w") as f:
#         f.write("Eref;Enn\n")
#         for i in range(data_count):
#             print(i)
#             data = get_data(i)
#             feed_dict = fill_feed_dict(data)
#
#             # calculate average prediction of ensemble
#             Eavg = 0
#             Favg = 0
#             Davg = 0
#             Qavg = 0
#             num = 1
#             for checkpoint in checkpoints:
#                 nn.restore(sess, checkpoint)
#                 Etmp, Ftmp, Dtmp, Qtmp = sess.run([energy, forces, D, Qtot], feed_dict=feed_dict)
#                 Eavg += (Etmp - Eavg) / num
#                 Favg += (Ftmp - Favg) / num
#                 Davg += (Dtmp - Davg) / num
#                 Qavg += (Qtmp - Qavg) / num
#                 num += 1
#
#                 # calculate errors
#             emae_tmp, emse_tmp = calculate_mae_mse(Eavg, data["E"])
#             fmae_tmp, fmse_tmp = calculate_mae_mse(Favg, data["F"])
#             dmae_tmp, dmse_tmp = calculate_mae_mse(Davg, data["D"])
#             qmae_tmp, qmse_tmp = calculate_mae_mse(Qavg, data["Q"])
#
#             # add to average errors
#             emae_avg += (emae_tmp - emae_avg) / (i + 1)
#             emse_avg += (emse_tmp - emse_avg) / (i + 1)
#             fmae_avg += (fmae_tmp - fmae_avg) / (i + 1)
#             fmse_avg += (fmse_tmp - fmse_avg) / (i + 1)
#             qmae_avg += (qmae_tmp - qmae_avg) / (i + 1)
#             qmse_avg += (qmse_tmp - qmse_avg) / (i + 1)
#             dmae_avg += (dmae_tmp - dmae_avg) / (i + 1)
#             dmse_avg += (dmse_tmp - dmse_avg) / (i + 1)
#             f.write(str(data["E"][0]) + ";" + str(Eavg) + "\n")
#
#     print("EMAE :", emae_avg)
#     print("ERMSE:", np.sqrt(emse_avg))
#     print("FMAE :", fmae_avg)
#     print("FRMSE:", np.sqrt(fmse_avg))
#     print("QMAE :", qmae_avg)
#     print("QRMSE:", np.sqrt(qmse_avg))
#     print("DMAE :", dmae_avg)
#     print("DRMSE:", np.sqrt(dmse_avg))
#
#
# # helper function to print errors
# def print_errors(sess, get_data, data_count, data_name, checkpoints):
#     print("\n" + data_name + " (" + str(data_count) + "):")
#
#     emse_avg = {}
#     emae_avg = {}
#     fmse_avg = {}
#     fmae_avg = {}
#     qmse_avg = {}
#     qmae_avg = {}
#     dmse_avg = {}
#     dmae_avg = {}
#     for checkpoint in checkpoints:
#         emse_avg[checkpoint] = 0.0
#         emae_avg[checkpoint] = 0.0
#         fmse_avg[checkpoint] = 0.0
#         fmae_avg[checkpoint] = 0.0
#         qmse_avg[checkpoint] = 0.0
#         qmae_avg[checkpoint] = 0.0
#         dmse_avg[checkpoint] = 0.0
#         dmae_avg[checkpoint] = 0.0
#     emse_avg["ensemble"] = 0.0
#     emae_avg["ensemble"] = 0.0
#     fmse_avg["ensemble"] = 0.0
#     fmae_avg["ensemble"] = 0.0
#     qmse_avg["ensemble"] = 0.0
#     qmae_avg["ensemble"] = 0.0
#     dmse_avg["ensemble"] = 0.0
#     dmae_avg["ensemble"] = 0.0
#
#     # in case we have only one checkpoint, the nn does not need to be restored
#     if len(checkpoints) == 1:
#         nn.restore(sess, checkpoints[0])
#
#     with open(data_name + ".dat", "w") as f:
#         # with open("findoutliers.csv", "w") as f:
#         # f.write("Eref;Enn\n")
#
#         for i in range(data_count):
#             print(i)
#             data = get_data(i)
#             feed_dict = fill_feed_dict(data)
#
#             # calculate average prediction of ensemble
#             Eavg = 0
#             Favg = 0
#             Davg = 0
#             Qavg = 0
#             num = 1
#             for checkpoint in checkpoints:
#                 if len(checkpoints) > 1:
#                     nn.restore(sess, checkpoint)
#                 Etmp, Ftmp, Dtmp, Qtmp = sess.run([energy, forces, D, Qtot], feed_dict=feed_dict)
#                 # compute errors for this checkpoint
#                 emae_tmp, emse_tmp = calculate_mae_mse(Etmp, data["E"])
#                 fmae_tmp, fmse_tmp = calculate_mae_mse(Ftmp, data["F"])
#                 dmae_tmp, dmse_tmp = calculate_mae_mse(Dtmp, data["D"])
#                 qmae_tmp, qmse_tmp = calculate_mae_mse(Qtmp, data["Q"])
#                 # add to average errors for this checkpoint
#                 emae_avg[checkpoint] += (emae_tmp - emae_avg[checkpoint]) / (i + 1)
#                 emse_avg[checkpoint] += (emse_tmp - emse_avg[checkpoint]) / (i + 1)
#                 fmae_avg[checkpoint] += (fmae_tmp - fmae_avg[checkpoint]) / (i + 1)
#                 fmse_avg[checkpoint] += (fmse_tmp - fmse_avg[checkpoint]) / (i + 1)
#                 qmae_avg[checkpoint] += (qmae_tmp - qmae_avg[checkpoint]) / (i + 1)
#                 qmse_avg[checkpoint] += (qmse_tmp - qmse_avg[checkpoint]) / (i + 1)
#                 dmae_avg[checkpoint] += (dmae_tmp - dmae_avg[checkpoint]) / (i + 1)
#                 dmse_avg[checkpoint] += (dmse_tmp - dmse_avg[checkpoint]) / (i + 1)
#                 # update ensemble predictions
#                 Eavg += (Etmp - Eavg) / num
#                 Favg += (Ftmp - Favg) / num
#                 Davg += (Dtmp - Davg) / num
#                 Qavg += (Qtmp - Qavg) / num
#                 num += 1
#
#                 # calculate errors
#             emae_tmp, emse_tmp = calculate_mae_mse(Eavg, data["E"])
#             fmae_tmp, fmse_tmp = calculate_mae_mse(Favg, data["F"])
#             dmae_tmp, dmse_tmp = calculate_mae_mse(Davg, data["D"])
#             qmae_tmp, qmse_tmp = calculate_mae_mse(Qavg, data["Q"])
#
#             # add to average errors of ensemble
#             emae_avg["ensemble"] += (emae_tmp - emae_avg["ensemble"]) / (i + 1)
#             emse_avg["ensemble"] += (emse_tmp - emse_avg["ensemble"]) / (i + 1)
#             fmae_avg["ensemble"] += (fmae_tmp - fmae_avg["ensemble"]) / (i + 1)
#             fmse_avg["ensemble"] += (fmse_tmp - fmse_avg["ensemble"]) / (i + 1)
#             qmae_avg["ensemble"] += (qmae_tmp - qmae_avg["ensemble"]) / (i + 1)
#             qmse_avg["ensemble"] += (qmse_tmp - qmse_avg["ensemble"]) / (i + 1)
#             dmae_avg["ensemble"] += (dmae_tmp - dmae_avg["ensemble"]) / (i + 1)
#             dmse_avg["ensemble"] += (dmse_tmp - dmse_avg["ensemble"]) / (i + 1)
#
#             f.write(str(data["E"][0]) + "  " + str(Eavg) + "\n")
#             # f.write(str(i)+";"+str(data["E"][0])+";"+str(Eavg)+";" + str(emae_tmp) + "\n")  # find structures with highest errors
#             # print_xyz(get_data, [8812,16855,14135,17431], "outliers")                       # when found use index to extract structures
#     # print results
#     print("RESULTS:\n")
#     ############################################## UNCOMMENT FOR THE RESULTS OF AN ENSEMBLE
#     # checkpoints.append("ensemble")
#     for checkpoint in checkpoints:
#         print(checkpoint)
#         if not np.isnan(emae_avg[checkpoint]):
#             print("EMAE :", emae_avg[checkpoint])
#         if not np.isnan(emse_avg[checkpoint]):
#             print("ERMSE:", np.sqrt(emse_avg[checkpoint]))
#         if not np.isnan(fmae_avg[checkpoint]):
#             print("FMAE :", fmae_avg[checkpoint])
#         if not np.isnan(fmse_avg[checkpoint]):
#             print("FRMSE:", np.sqrt(fmse_avg[checkpoint]))
#         if not np.isnan(fmae_avg[checkpoint]):
#             print("QMAE :", qmae_avg[checkpoint])
#         if not np.isnan(qmse_avg[checkpoint]):
#             print("QRMSE:", np.sqrt(qmse_avg[checkpoint]))
#         if not np.isnan(dmae_avg[checkpoint]):
#             print("DMAE :", dmae_avg[checkpoint])
#         if not np.isnan(dmse_avg[checkpoint]):
#             print("DRMSE:", np.sqrt(dmse_avg[checkpoint]))
#         print()
#
#
# # prints xyz files of structures in the data set given the index list
# def print_xyz(get_data, index_list, data_name):
#     if not os.path.exists(data_name):
#         os.makedirs(data_name)
#     for i in index_list:
#         data = get_data(i)
#         Z = data["Z"]
#         R = data["R"]
#         Qref = data["Q"]
#         Dref = data["D"]
#         N = np.shape(Z)[0]
#         with open(data_name + "/" + str(i) + ".xyz", "w") as f:
#             f.write(str(N) + "\n")
#             f.write(str(Eref[0]) + " " + str(Qref) + " " + str(Dref[0]) + "\n")
#             for a in range(N):
#                 f.write(str(Z[a]) + " " + str(R[a][0])
#                         + " " + str(R[a][1])
#                         + " " + str(R[a][2]) + "\n")
#
#             # create tensorflow session
#
#
# with tf.Session() as sess:
#     # generate dataset with atomic quantities
#     # generate_dataset_with_atomic_quantities(sess, data, checkpoints)
#
#     # calculate errors on test data
#     print_errors(sess, data_provider.get_test_data, data_provider.ntest, "error_qm9-alone", checkpoints)



