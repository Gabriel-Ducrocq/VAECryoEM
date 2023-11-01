import os

import protein.main
import yaml
import torch
import utils_data as utils
import argparse
import numpy as np
from tqdm import tqdm
from Bio.PDB import PDBParser
import Bio.PDB as bpdb
import matplotlib.pyplot as plt
from pytorch3d.transforms import axis_angle_to_matrix

class ResSelect(bpdb.Select):
    def accept_residue(self, res):
        if res.get_resname() == "LBV":
            return False
        else:
            return True


parser_arg = argparse.ArgumentParser()
parser_arg.add_argument('--folder_experiment', type=str, required=True)
args = parser_arg.parse_args()
folder_experiment = args.folder_experiment


with open(f"{folder_experiment}/parameters.yaml", "r") as file:
    experiment_settings = yaml.safe_load(file)


parser = PDBParser(PERMISSIVE=0)
base_structure = parser.get_structure("A", experiment_settings["base_structure_path"])
center_vector = utils.compute_center_of_mass(base_structure)
io = bpdb.PDBIO()
## Save structure while removing biliverdin
io.set_structure(base_structure)
io.save(f"{folder_experiment}temp.pdb", ResSelect())

#trans = np.array([ 9.333321  , -0.45512402, -9.019102  ], dtype=np.float32)
#trans[1] =-6
#utils.center_protein(base_structure, -trans)
#utils.save_structure(base_structure, f"{folder_experiment}/base_structure_shifted_by_12_angstrom.pdb")
translation_per_residues = np.load(f"{folder_experiment}all_translation_per_residue.npy")
rotation_per_residues = np.load(f"{folder_experiment}all_rotations_per_residue.npy")
temp_structure = parser.get_structure("A", f"{folder_experiment}temp.pdb")
utils.center_protein(temp_structure, center_vector[0])
#protein.main.rotate_residues(temp_structure, rotation_per_residues[403], np.eye(3, 3))
protein.main.translate_residues(temp_structure, translation_per_residues[403])
utils.center_protein(temp_structure, -center_vector[0])
utils.save_structure(temp_structure, f"{folder_experiment}only_translation_based_structure_404.pdb")
#utils.save_structure(temp_structure, f"{folder_experiment}rotation_and_translation_based_structure_404.pdb")




