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
parser_arg.add_argument('--folder_experiment', type=str, required=True)
parser_arg.add_argument('--folder_structures', type=str, required=True)
args = parser_arg.parse_args()
folder_experiment = args.folder_experiment
folder_structures = args.folder_structures


#Get all the structure and sort their names to have them in the right order.
structures = [folder_structures + path for path in os.listdir(folder_structures) if ".pdb" in path]
indexes = [int(name.split("/")[-1].split(".")[0].split("_")[-1]) for name in structures]
sorted_structures = [struct for _, struct in sorted(zip(indexes, structures))]


#Finds the pdb file that will serve as a base structure
with open(folder_experiment + "parameters.yaml", "r") as file:
    experiments_settings = yaml.safe_load(file)

centering_structure_path = experiments_settings["centering_structure_path"]
parser = PDBParser(PERMISSIVE=0)
centering_structure = parser.get_structure("A", centering_structure_path)

#Create poses:
N_images = experiments_settings["N_images"]
axis_rotation = torch.randn((N_images, 3))
norm_axis = torch.sqrt(torch.sum(axis_rotation**2, dim=-1))
normalized_axis = axis_rotation/norm_axis[:, None]
print("Min norm of rotation axis", torch.min(torch.sqrt(torch.sum(normalized_axis**2, dim=-1))))
print("Max norm of rotation axis", torch.max(torch.sqrt(torch.sum(normalized_axis**2, dim=-1))))

angle_rotation = torch.rand((N_images,1))*torch.pi
plt.hist(angle_rotation[:, 0].detach().numpy())
plt.show()

axis_angle = normalized_axis*angle_rotation
poses = axis_angle_to_matrix(axis_angle)
poses_translation = torch.zeros((N_images, 3))
poses_translation[:, :2] = 20*torch.rand((N_images, 2)) - 10

poses_py = poses.detach().numpy()
poses_translation_py = poses_translation.detach().numpy()


print("Min translation", torch.min(poses_translation))
print("Max translation", torch.max(poses_translation))

#Finding the center of mass to center the protein
center_vector = utils.compute_center_of_mass(centering_structure)
with warnings.catch_warnings():
    warnings.simplefilter('ignore', BiopythonWarning)
    for i, structure in tqdm(enumerate(sorted_structures)):
        posed_structure = utils.compute_poses(structure, poses_py[i], poses_translation_py[i], center_vector)
        utils.save_structure(posed_structure, f"{folder_experiment}posed_structures/structure_{i+1}.pdb")
        np.save(f"{folder_experiment}posed_structures/poses.npy", poses_py)
        np.save(f"{folder_experiment}posed_structures/poses_translation.npy", poses_translation_py)
        torch.save(poses, f"{folder_experiment}poses")
        torch.save(poses_translation, f"{folder_experiment}poses_translation")




