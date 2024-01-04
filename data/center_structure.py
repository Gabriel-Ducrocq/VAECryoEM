import os
import yaml
import torch
import warnings
import utils_data as utils
import argparse
from Bio import BiopythonWarning
import numpy as np
from tqdm import tqdm
from Bio.PDB import PDBParser
import matplotlib.pyplot as plt
from pytorch3d.transforms import axis_angle_to_matrix

parser_arg = argparse.ArgumentParser()
parser_arg.add_argument('--structure', type=str, required=True)
parser_arg.add_argument('--centering_structure', type=str, required=True)
parser_arg.add_argument('--o', type=str, required=True)
args = parser_arg.parse_args()
centering_structure_path = args.centering_structure
structure_path = args.structure
output = args.o
parser = PDBParser(PERMISSIVE=0)
centering_structure = parser.get_structure("A", centering_structure_path)
structure = parser.get_structure("A", structure_path)
center_vector = utils.compute_center_of_mass(centering_structure)
print(center_vector)

centered_structure = utils.center_protein(structure, center_vector[0])
#centered_structure = utils.center_protein(centering_structure, -np.array((95, 95, 95)))
utils.save_structure(centered_structure, output)
