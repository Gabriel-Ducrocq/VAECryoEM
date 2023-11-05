import sys
import os

import protein.main

path = os.path.abspath("model")
sys.path.append(path)
import torch
import yaml
import utils
import argparse
import numpy as np
from tqdm import tqdm
import Bio.PDB as bpdb
from Bio.PDB import PDBParser
from Bio.PDB import PDBParser
from renderer import Renderer
from dataset import ImageDataSet
from torch.utils.data import DataLoader
from protein.main import rotate_residues
from protein.main import translate_residues

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

rotations_per_residues = np.load(f"{folder_experiment}all_rotations_per_residue.npy")
translations_per_residues = np.load(f"{folder_experiment}all_translation_per_residue.npy")
with open(f"{folder_experiment}/parameters.yaml", "r") as file:
    experiment_settings = yaml.safe_load(file)


parser = PDBParser()
N_images = rotations_per_residues.shape[0]
for i in tqdm(range(N_images)):
    io = bpdb.PDBIO()
    base_structure = utils.read_pdb(experiment_settings["base_structure_path"])
    io.set_structure(base_structure)
    ## Save structure while removing biliverdin
    io.save(f"{folder_experiment}predicted_structures/predicted_test_{i+1}.pdb", ResSelect())
    structure = utils.read_pdb(f"{folder_experiment}predicted_structures/predicted_test_{i+1}.pdb")
    center_of_mass = utils.compute_center_of_mass(structure)

    #Center before applying the transformations, to be consistent with the VAE.
    structure = utils.center_protein(structure, center_of_mass[0])
    rot_per_residue = rotations_per_residues[i]
    trans_per_residue = translations_per_residues[i]
    rotate_residues(structure, rot_per_residue, np.eye(3,3))
    translate_residues(structure, trans_per_residue)

    #Decenter before writing, so that we keep the same centering as the dataset.
    structure = utils.center_protein(structure, -center_of_mass[0])
    io = bpdb.PDBIO()
    io.set_structure(structure)
    io.save(f"{folder_experiment}predicted_structures/predicted_test_{i+1}.pdb")


