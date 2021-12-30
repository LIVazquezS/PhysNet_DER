#imports
import argparse
from ase import Atoms
import numpy as np
from ase.io import read, write
from calculator import PhysNetCalculator

#parse command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser._action_groups.pop()
required = parser.add_argument_group("required arguments")
required.add_argument("-i", "--input",   type=str,   help="input xyz",  required=True)
optional = parser.add_argument_group("optional arguments")
optional.add_argument("--charge",  type=float, help="total charge", default=0.0)
from os.path import splitext

args = parser.parse_args()
filename, extension = splitext(args.input)

#read input file
atoms = read(args.input)

#setup calculator (which will be used to describe the atomic interactions)
calc = PhysNetCalculator(
    checkpoint="best_model.pt",
    atoms=atoms,
    charge=args.charge,
    config='input.inp')

#'attach' calculator to atoms object
atoms.set_calculator(calc)


#print potential energy (to scalar to display more digits)

e,var,sigma2 = atoms.get_potential_energy()

e_final = np.asscalar(e)
var_final = np.asscalar(var)

print("Potential energy: %.8f eV" % e_final)
print("Variance: %.8f eV" % var_final)