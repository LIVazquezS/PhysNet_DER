#!/usr/bin/env python3
from ase.units import *
import re
from ase.io import read, write
import os
import numpy as np
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser._action_groups.pop()
required = parser.add_argument_group("required arguments")
required.add_argument("-f", "--folder",   type=str,   help="folder with .xyz files",  required=True)
required.add_argument("-o", "--output",  type=str,   help="output npz",     required=True)
args = parser.parse_args()
print('Your file with .xyz files is: ', args.folder)
print('Your output file is: ', args.output)
folder = args.folder
output = args.output
file_list = []
for path, dirs, files in os.walk(folder):
    for file in files:
        if file.endswith(".xyz"):
            file_list.append(path+"/"+file)

Nmax = 30  #needs to be adapted(max number of atoms)

num = len(file_list)
N = np.zeros([num], dtype=int) #Number of atoms
E = np.zeros([num], dtype=float) #Potential energy with respect to atoms
Q = np.zeros([num], dtype=float) #Total charge
D = np.zeros([num, 3], dtype=float) #Dipole 
Z = np.zeros([num, Nmax], dtype=int) #Nuclear charges/atomic numbers of nuclei
R = np.zeros([num, Nmax, 3], dtype=float) #Cartesian coordinates of nuclei
F = np.zeros([num, Nmax, 3], dtype=float) #Forces

#reference B3LYP energies for the single H, C, N, O, F atoms in Ha. The corresponding value
#will be subtracted per corresponding atom in the molecule
Eref = np.zeros([10], dtype=float)
Eref[1] = -0.5002730
Eref[6] = -37.846771
Eref[7] = -54.583861
Eref[8] = -75.064579
Eref[9] = -99.718730

index = 0
for file in tqdm(file_list):
    # open file and read contents
    with open(file, "r") as f:
        contents = f.read().splitlines()

    #search for CARTESIAN COORDINATES:
    Ntmp = int(contents[0])
    Ztmp = []
    Rtmp = []
    Qatom = []
    for line in contents[2:2+Ntmp]:
        l, x, y, z, q = line.split()
        if l == 'H':
            Ztmp.append(1)
        elif l == 'C':
            Ztmp.append(6)
        elif l == 'N':
            Ztmp.append(7)
        elif l == 'O':
            Ztmp.append(8)
        elif l == 'F':
            Ztmp.append(9)
        else:
            print("UNKNOWN LABEL", l)
            quit()    
        Rtmp.append([float(x.replace('*^','e')), float(y.replace('*^','e')), float(z.replace('*^','e'))])
        Qatom.append(float(q.replace('*^','e')))
          
    #Total charge    
    Qtmp = int(np.sum(Qatom))	
    
    #search for FINAL SINGLE POINT ENERGY in Ha:
    linenumberE = contents[1].split()
    Etmp = float(linenumberE[12].replace('*^','e'))
    
    #subtract asymptotics
    for z in Ztmp:
        Etmp -= Eref[z] 
    Dx = 0
    Dy = 0
    Dz = 0
    #dipole moment 
    for k in range(0,Ntmp):
        Dxa = Qatom[k]*(Rtmp[k][0])
        Dya = Qatom[k]*(Rtmp[k][1])
        Dza = Qatom[k]*(Rtmp[k][2])
        Dx += Dxa
        Dy += Dya
        Dz += Dza
    Dtmp = [Dx, Dy, Dz]
	
    
    
    N[index] = Ntmp
    E[index] = Etmp
    Q[index] = Qtmp
    D[index,:] = np.asarray(Dtmp)
    Z[index,:Ntmp] = np.asarray(Ztmp)
    R[index,:Ntmp,:] = np.asarray(Rtmp)
    #F[index,:Ntmp,:] = np.asarray(Ftmp)

    index += 1


#unit conversion
E *= Hartree
D *= Debye
#F *= Hartree/Bohr

np.savez_compressed(output, N=N, E=E, Q=Q, D=D, Z=Z, R=R, F=F)



