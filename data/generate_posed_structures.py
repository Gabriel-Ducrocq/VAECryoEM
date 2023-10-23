import os
import torch
import utils
import numpy as np
from Bio.PDB import PDBParser

parser = PDBParser(PERMISSIVE=0)

poses = torch.load("../../VAEProtein/data/vaeContinuousMD/training_rotations_matrices")
poses = poses.numpy()
structures = ["../../VAEProtein/data/MD_dataset/" + path for path in os.listdir("../../VAEProtein/data/MD_dataset/") if ".pdb" in path]

indexes = [int(name.split("/")[-1].split(".")[0].split("_")[-1]) for name in structures]
sorted_structures = [struct for _, struct in sorted(zip(indexes, structures))]

center_vector = utils.compute_center_of_mass(parser.get_structure("A", sorted_structures[0]))
posed_structures = utils.compute_poses(sorted_structures, poses, center_vector)
utils.save_structures(posed_structures, "MD_simulation/posed_structures/")
np.save("MD_simulation/poses.npy", poses)
