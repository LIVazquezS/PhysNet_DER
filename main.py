import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
from NeuralNet import PhysNet
from DataContainer import DataContainer
from layers.RBFLayer import RBFLayer
from NeuralNet import gather_nd

data_tot = DataContainer('data/sn2_reactions.npz', 362167, 45000,
     100, 20, 0)
#

train_batches,N_train_batches = data_tot.get_train_batches()

model = PhysNet()
def get_indices(Nref):
    # Get indices pointing to batch image
    # For some reason torch does not make repeatition for float
    batch_seg = torch.arange(0, Nref.size()[0]).repeat_interleave(Nref.type(torch.int32))

    # Initiate auxiliary parameter
    Nref_tot = torch.tensor(0, dtype=torch.int32)

    # Indices pointing to atom at each batch image
    idx = torch.arange(end=Nref[0], dtype=torch.int32)
    # Indices for atom pairs ij - Atom i
    idx_i = idx.repeat(int(Nref.numpy()[0]) - 1) + Nref_tot
    # Indices for atom pairs ij - Atom j
    idx_j = torch.roll(idx, -1, dims=0) + Nref_tot
    for Na in torch.arange(2, Nref[0]):
        idx_j = torch.concat(
            [idx_j, torch.roll(idx, int(-Na.numpy()), dims=0) + Nref_tot],
            dim=0)

    # Increment auxiliary parameter
    Nref_tot = Nref_tot + Nref[0]

    # Complete indices arrays
    for Nref_a in Nref[1:]:

        rng_a = torch.arange(end=Nref_a)

        idx = torch.concat([idx, rng_a], axis=0)
        idx_i = torch.concat(
            [idx_i, rng_a.repeat(int(Nref_a.numpy()) - 1) + Nref_tot],
            dim=0)
        for Na in torch.arange(1, Nref_a):
            idx_j = torch.concat(
                [idx_j, torch.roll(rng_a, int(-Na.numpy()), dims=0) + Nref_tot],
                dim=0)

        # Increment auxiliary parameter
        Nref_tot = Nref_tot + Nref_a

    # Combine indices for batch image and respective atoms
    idx = torch.stack([batch_seg, idx], dim=1)
    idx.type(torch.int64)
    idx_i.type(torch.int64)
    idx_j.type(torch.int64)
    return idx, idx_i, idx_j, batch_seg

optimizer = torch.optim.Adam(params=model.parameters(),lr=0.001,
                             weight_decay=0.1,amsgrad=True)

for ib,batch in enumerate(train_batches):
    N_t, Z_t, R_t, Eref_t, Earef_t, Fref_t, Qref_t, Qaref_t, Dref_t = batch
    print(Dref_t.shape)
    # Get indices
    idx_t, idx_i_t, idx_j_t, batch_seg_t = get_indices(N_t)
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
    D_t = torch.zeros(u_bst,3).index_add(0, batch_seg_t.type(torch.int64), QR_t)



    print(D_t.shape)
    # if torch.count_nonzero(Ea_t) != 0:
    #     Ea_t = gather_nd(Ea_t,idx_t)
    # else:
    #     Ea_t = torch.zeros(energy_t.shape)


